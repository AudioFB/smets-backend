import os
import uuid
import traceback
import shutil
import threading
import json
import hashlib
from argparse import Namespace
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS

from separate import SeperateDemucs, SeperateMDX, SeperateMDXC
from lib_v5.vr_network.model_param_init import ModelParameters

# --- DEFINIÇÃO MANUAL DE CONSTANTES ---
DEMUCS_ARCH_TYPE = 'Demucs'
MDX_ARCH_TYPE = 'MDX-Net'
DEMUCS_V4 = 'v4'
ALL_STEMS = 'All Stems'
DEFAULT = 'Default'
WAV = 'WAV'
WAV_TYPE_16 = 'PCM_16'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
DEMUCS_2_SOURCE_MAPPER = {'vocals': 0, 'instrumental': 1}
DEMUCS_4_SOURCE_MAPPER = {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3}
DEMUCS_6_SOURCE_MAPPER = {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3, 'guitar': 4, 'piano': 5}

# --- Configuração da Aplicação ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://audiofb.com"}})

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

JOB_STATUS = {}
job_status_lock = threading.Lock()

# --- FUNÇÕES AUXILIARES ---
def get_model_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(-10000 * 1024, 2)
            return hashlib.md5(f.read()).hexdigest()
    except:
        return hashlib.md5(open(model_path, 'rb').read()).hexdigest()

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

MDX_HASH_JSON = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', 'model_data', 'model_data.json')
MDX_MODEL_PARAMS = load_json_data(MDX_HASH_JSON)

def get_models_from_dir(directory):
    models = []
    if not os.path.isdir(directory):
        print(f"Diretório não encontrado: {directory}")
        return models
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.onnx', '.ckpt', '.yaml', '.gz', '.th')):
                models.append(os.path.splitext(file)[0])
    return sorted(list(set(models)))

@app.route('/models', methods=['GET'])
def get_models():
    try:
        mdx_models = get_models_from_dir(os.path.join(MODELS_FOLDER, 'MDX_Net_Models'))
        demucs_models = get_models_from_dir(os.path.join(MODELS_FOLDER, 'Demucs_Models'))
        
        models = {
            'MDX-Net': mdx_models,
            'Demucs': demucs_models
        }
        return jsonify(models)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Erro ao listar modelos"}), 500

def run_separation_in_background(job_id, model_data, process_data, app_context):
    with app_context:
        try:
            separator = None
            if model_data.process_method == DEMUCS_ARCH_TYPE:
                separator = SeperateDemucs(model_data=model_data, process_data=process_data)
            elif model_data.process_method == MDX_ARCH_TYPE:
                if getattr(model_data, 'is_mdx_c', False):
                    separator = SeperateMDXC(model_data=model_data, process_data=process_data)
                else:
                    separator = SeperateMDX(model_data=model_data, process_data=process_data)

            if separator:
                separator.seperate()
                result_files = os.listdir(process_data['export_path'])
                with job_status_lock:
                    JOB_STATUS[job_id]['status'] = 'complete'
                    JOB_STATUS[job_id]['progress'] = 100
                    JOB_STATUS[job_id]['files'] = result_files
            else:
                 raise ValueError("Método de processamento não suportado")
        except Exception as e:
            traceback.print_exc()
            with job_status_lock:
                JOB_STATUS[job_id]['status'] = 'error'
                JOB_STATUS[job_id]['error_message'] = str(e)

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files: return jsonify({"error": "Nenhum arquivo"}), 400
    file = request.files['audio_file']
    if file.filename == '': return jsonify({"error": "Nome inválido"}), 400
    
    model_name_from_request = request.form.get('model_name')
    process_method_from_request = request.form.get('process_method')
    
    if not model_name_from_request or not process_method_from_request:
        return jsonify({"error": "Modelo ou método de processamento não especificado"}), 400

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
    file.save(input_path)
    output_path_for_job = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(output_path_for_job, exist_ok=True)

    def update_progress(base, step=0):
        progress = int((base + step) * 100)
        with job_status_lock:
            if job_id in JOB_STATUS and progress > JOB_STATUS[job_id].get('progress', 0):
                JOB_STATUS[job_id]['progress'] = min(progress, 99)
    
    process_data = {
        'audio_file': input_path, 'audio_file_base': os.path.splitext(os.path.basename(input_path))[0],
        'export_path': output_path_for_job, 'set_progress_bar': update_progress,
        'write_to_console': lambda text, base_text="": None, 'process_iteration': lambda: None,
        'cached_source_callback': lambda *args, **kwargs: (None, None),
        'cached_model_source_holder': lambda *args, **kwargs: None,
        'is_ensemble_master': False, 'is_4_stem_ensemble': False, 'list_all_models': []
    }

    params = {}
    model_path = ""
    
    if process_method_from_request == MDX_ARCH_TYPE:
        onnx_path = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{model_name_from_request}.onnx')
        ckpt_path = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{model_name_from_request}.ckpt')
        if os.path.exists(onnx_path):
            model_path = onnx_path
            params['is_mdx_ckpt'] = False
        elif os.path.exists(ckpt_path):
            model_path = ckpt_path
            params['is_mdx_ckpt'] = True
        
        model_hash = get_model_hash(model_path)
        model_params_json = MDX_MODEL_PARAMS.get(model_hash, {})
        params['compensate'] = model_params_json.get('compensate', 1.035)
        params['mdx_dim_f_set'] = model_params_json.get('mdx_dim_f_set')
        params['mdx_dim_t_set'] = model_params_json.get('mdx_dim_t_set')
        params['mdx_n_fft_scale_set'] = model_params_json.get('mdx_n_fft_scale_set')
        params['primary_stem'] = model_params_json.get('primary_stem', VOCAL_STEM)

    elif process_method_from_request == DEMUCS_ARCH_TYPE:
        demucs_base_path = os.path.join(MODELS_FOLDER, 'Demucs_Models')
        if os.path.exists(os.path.join(demucs_base_path, 'v3_v4_repo', f'{model_name_from_request}.yaml')):
            model_path = os.path.join(demucs_base_path, 'v3_v4_repo', f'{model_name_from_request}.yaml')
        else:
            model_path = os.path.join(demucs_base_path, f'{model_name_from_request}.ckpt')
        
        if '6s' in model_name_from_request:
            params['demucs_stem_count'], params['demucs_source_map'], params['demucs_source_list'] = 6, DEMUCS_6_SOURCE_MAPPER, list(DEMUCS_6_SOURCE_MAPPER.keys())
        else:
            params['demucs_stem_count'], params['demucs_source_map'], params['demucs_source_list'] = (4, DEMUCS_4_SOURCE_MAPPER, list(DEMUCS_4_SOURCE_MAPPER.keys())) if '4s' in model_name_from_request else (2, DEMUCS_2_SOURCE_MAPPER, list(DEMUCS_2_SOURCE_MAPPER.keys()))

    model_data = Namespace(
        **{
            # --- CORREÇÃO: Bloco completo de atributos padrão restaurado ---
            **dict(
                is_mdx_ckpt=False, is_tta=False, is_post_process=False, is_high_end_process='none', 
                post_process_threshold=0.1, aggression_setting=0.1, batch_size=4, window_size=512, 
                is_denoise=False, is_denoise_model=False, is_mdx_c_seg_def=False, mdx_batch_size=1, 
                compensate=1.035, mdx_segment_size=256, mdx_dim_f_set=None, mdx_dim_t_set=None, 
                mdx_n_fft_scale_set=None, chunks=0, margin=44100, demucs_version=DEMUCS_V4, 
                segment=DEFAULT, shifts=2, overlap=0.25, is_split_mode=True, is_chunk_demucs=True, 
                demucs_stems=ALL_STEMS, is_demucs_combine_stems=True, demucs_source_list=[], 
                demucs_source_map={}, demucs_stem_count=0, is_gpu_conversion=-1, device_set=DEFAULT, 
                is_use_opencl=False, wav_type_set=WAV_TYPE_16, mp3_bit_set='320k', save_format=WAV, 
                is_normalization=False, is_primary_stem_only=False, is_secondary_stem_only=False, 
                is_ensemble_mode=False, is_pitch_change=False, semitone_shift=0, 
                is_match_frequency_pitch=False, is_secondary_model_activated=False, 
                secondary_model=None, pre_proc_model=None, is_secondary_model=False, overlap_mdx=0.5, 
                overlap_mdx23=8, is_mdx_combine_stems=False, is_mdx_c=False, mdx_c_configs=None, 
                mdxnet_stem_select=VOCAL_STEM, mixer_path='lib_v5/mixer.ckpt', model_samplerate=44100, 
                model_capacity=(64, 128), is_vr_51_model=False, is_pre_proc_model=False, 
                primary_model_primary_stem=VOCAL_STEM, primary_stem_native=VOCAL_STEM, 
                primary_stem=VOCAL_STEM, secondary_stem=INST_STEM, is_invert_spec=False, 
                is_deverb_vocals=False, is_mixer_mode=False, secondary_model_scale=0.5, 
                is_demucs_pre_proc_model_inst_mix=False, DENOISER_MODEL=None, DEVERBER_MODEL=None, 
                vocal_split_model=None, is_vocal_split_model=False, is_save_inst_vocal_splitter=False, 
                is_inst_only_voc_splitter=False, is_karaoke=False, is_bv_model=False, 
                bv_model_rebalance=0.0, is_sec_bv_rebalance=False, deverb_vocal_opt=None, 
                is_save_vocal_only=False, secondary_model_4_stem=[None]*4, 
                secondary_model_4_stem_scale=[0.5]*4, ensemble_primary_stem=VOCAL_STEM, 
                is_multi_stem_ensemble=False
            ), 
            **params, 
            **dict(
                process_method=process_method_from_request, model_path=model_path,
                model_name=model_name_from_request, model_basename=model_name_from_request
            )
        }
    )
    
    with job_status_lock:
        JOB_STATUS[job_id] = {'status': 'processing', 'progress': 0, 'files': []}

    thread = threading.Thread(target=run_separation_in_background, args=(job_id, model_data, process_data, app.app_context()))
    thread.start()

    return jsonify({"job_id": job_id}), 202

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    with job_status_lock:
        job = JOB_STATUS.get(job_id)
        return jsonify(job) if job else (jsonify({"status": "not_found"}), 404)

@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    directory = os.path.join(OUTPUT_FOLDER, job_id)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    data = request.get_json(force=True)
    job_id = data.get('job_id')
    if not job_id: return jsonify({"error": "job_id não fornecido"}), 400
    try:
        with job_status_lock:
            if job_id in JOB_STATUS: del JOB_STATUS[job_id]
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        for f in os.listdir(UPLOAD_FOLDER):
            if f.startswith(job_id): os.remove(os.path.join(UPLOAD_FOLDER, f))
        return jsonify({"message": "Arquivos apagados."}), 200
    except Exception as e:
        return jsonify({"error": "Erro na limpeza"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
# run_separation.py
import os
import argparse
import requests
import uuid
import json
import hashlib
import boto3
from argparse import Namespace
import torch
from demucs.hdemucs import HDemucs
import zipfile
import traceback

torch.serialization.add_safe_globals([HDemucs])
from separate import SeperateDemucs, SeperateMDX

# --- CONSTANTES ---
DEMUCS_ARCH_TYPE = 'Demucs'
MDX_ARCH_TYPE = 'MDX-Net'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
DEFAULT = 'Default'
WAV = 'WAV'
WAV_TYPE_16 = 'PCM_16'
ALL_STEMS = 'All Stems'
DEMUCS_V4 = 'v4'

# --- CAMINHOS E CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# --- FUNÇÕES AUXILIARES ---
def get_model_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(-10000 * 1024, 2)
            return hashlib.md5(f.read()).hexdigest()
    except:
        return hashlib.md5(open(model_path, 'rb').read()).hexdigest()

MDX_HASH_JSON = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', 'model_data', 'model_data.json')
MDX_MODEL_PARAMS = {}
if os.path.exists(MDX_HASH_JSON):
    with open(MDX_HASH_JSON, 'r') as f:
        MDX_MODEL_PARAMS = json.load(f)

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---
def execute_separation(args):
    print(f"--- Processo de separação iniciado para o Job ID: {args.jobId} ---")

    work_dir = f"/tmp/{args.jobId}"
    input_path = os.path.join(work_dir, args.filename)
    output_folder = work_dir

    process_data = {
        'audio_file': input_path,
        'audio_file_base': os.path.splitext(os.path.basename(input_path))[0],
        'export_path': output_folder,
        'set_progress_bar': lambda *a, **k: None,
        'write_to_console': lambda text, base_text="": print(text),
        'process_iteration': lambda: None,
        'cached_source_callback': lambda *a, **k: (None, None),
        'cached_model_source_holder': lambda *a, **k: None,
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False,
        'list_all_models': []
    }

    params, model_path = {}, ""
    
    if args.process_method == MDX_ARCH_TYPE:
        onnx_path = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{args.model_name}.onnx')
        ckpt_path = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{args.model_name}.ckpt')
        if os.path.exists(onnx_path):
            model_path = onnx_path
            params['is_mdx_ckpt'] = False
        elif os.path.exists(ckpt_path):
            model_path = ckpt_path
            params['is_mdx_ckpt'] = True
        else:
            return {"error": f"Modelo MDX-Net não encontrado: {args.model_name}"}
        
        model_hash = get_model_hash(model_path)
        model_params_json = MDX_MODEL_PARAMS.get(model_hash, {})
        params.update(model_params_json)

    elif args.process_method == DEMUCS_ARCH_TYPE:
        model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', 'v3_v4_repo', f'{args.model_name}.yaml')
        if not os.path.exists(model_path):
             model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', f'{args.model_name}.ckpt')

        if '6s' in args.model_name:
             params['demucs_stem_count'] = 6
        elif 'htdemucs' in args.model_name:
            params['demucs_stem_count'] = 4
        else:
            params['demucs_stem_count'] = 2

    default_params = {
        'is_mdx_ckpt': False, 'is_tta': False, 'is_post_process': False, 'is_high_end_process': 'none', 
        'post_process_threshold': 0.1, 'aggression_setting': 0.1, 'batch_size': 4, 'window_size': 512, 
        'is_denoise': False, 'is_denoise_model': False, 'is_mdx_c_seg_def': False, 'mdx_batch_size': 1, 
        'compensate': 1.035, 'mdx_segment_size': 256, 'mdx_dim_f_set': None, 'mdx_dim_t_set': None, 
        'mdx_n_fft_scale_set': None, 'chunks': 0, 'margin': 44100, 'demucs_version': DEMUCS_V4, 
        'segment': DEFAULT, 'shifts': 2, 'overlap': 0.25, 'is_split_mode': True, 'is_chunk_demucs': True, 
        'demucs_stems': ALL_STEMS, 'is_demucs_combine_stems': True, 'demucs_source_list': [], 
        'demucs_source_map': {}, 'demucs_stem_count': 0, 'is_gpu_conversion': True, 'device_set': '0', 
        'is_use_opencl': False, 'wav_type_set': WAV_TYPE_16, 'mp3_bit_set': '320k', 'save_format': WAV, 
        'is_normalization': False, 'is_primary_stem_only': False, 'is_secondary_stem_only': False, 
        'is_ensemble_mode': False, 'is_pitch_change': False, 'semitone_shift': 0, 
        'is_match_frequency_pitch': False, 'is_secondary_model_activated': False, 
        'secondary_model': None, 'pre_proc_model': None, 'is_secondary_model': False, 'overlap_mdx': 0.5, 
        'overlap_mdx23': 8, 'is_mdx_combine_stems': False, 'is_mdx_c': False, 'mdx_c_configs': None, 
        'mdxnet_stem_select': VOCAL_STEM, 'mixer_path': 'lib_v5/mixer.ckpt', 'model_samplerate': 44100, 
        'model_capacity': (64, 128), 'is_vr_51_model': False, 'is_pre_proc_model': False, 
        'primary_model_primary_stem': VOCAL_STEM, 'primary_stem_native': VOCAL_STEM, 
        'primary_stem': VOCAL_STEM, 'secondary_stem': INST_STEM, 'is_invert_spec': False, 
        'is_deverb_vocals': False, 'is_mixer_mode': False, 'secondary_model_scale': 0.5, 
        'is_demucs_pre_proc_model_inst_mix': False, 'DENOISER_MODEL': None, 'DEVERBER_MODEL': None, 
        'vocal_split_model': None, 'is_vocal_split_model': False, 'is_save_inst_vocal_splitter': False, 
        'is_inst_only_voc_splitter': False, 'is_karaoke': False, 'is_bv_model': False, 
        'bv_model_rebalance': 0.0, 'is_sec_bv_rebalance': False, 'deverb_vocal_opt': None, 
        'is_save_vocal_only': False, 'secondary_model_4_stem': [None]*4, 
        'secondary_model_4_stem_scale': [0.5]*4, 'ensemble_primary_stem': VOCAL_STEM, 
        'is_multi_stem_ensemble': False
    }
    
    job_specific_params = {'process_method': args.process_method, 'model_path': model_path, 'model_name': args.model_name, 'model_basename': args.model_name}
    final_params = {**default_params, **params, **job_specific_params}
    model_data = Namespace(**final_params)

    try:
        separator = None
        if model_data.process_method == DEMUCS_ARCH_TYPE:
            separator = SeperateDemucs(model_data=model_data, process_data=process_data)
        elif model_data.process_method == MDX_ARCH_TYPE:
            separator = SeperateMDX(model_data=model_data, process_data=process_data)
        if separator:
            separator.seperate()
            print("\nSeparação concluída!")
        else:
            raise ValueError(f"Método de processamento não suportado: {model_data.process_method}")
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Erro na separação: {e}"}

    try:
        zip_filename_r2 = f"{args.jobId}-mixbusted.zip"
        zip_path_local = os.path.join(work_dir, zip_filename_r2)
        
        with zipfile.ZipFile(zip_path_local, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(output_folder):
                if file.lower().endswith(('.wav', '.mp3', '.flac')) and file != args.filename:
                    zipf.write(os.path.join(output_folder, file), os.path.basename(file))
        print("Compactação concluída.")

        s3 = boto3.client(
            service_name='s3', endpoint_url=os.environ.get('R2_ENDPOINT_URL'),
            aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
            region_name='auto'
        )
        
        bucket_name = os.environ.get('R2_BUCKET_NAME')
        public_domain = os.environ.get('R2_PUBLIC_DOMAIN')
        
        print(f"Enviando {zip_filename_r2} para o bucket R2: {bucket_name}")
        s3.upload_file(zip_path_local, bucket_name, zip_filename_r2)
        download_url = f"https://{public_domain}/{zip_filename_r2}"
        print(f"Upload para R2 concluído! URL de download: {download_url}")
        
        finish_url = f'{args.baseUrl}/mixbuster/finish_job.php'
        payload = {'jobId': args.jobId, 'downloadUrl': download_url, 'originalFilename': args.filename}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.post(finish_url, data=payload, headers=headers)
        response.raise_for_status()
        
        print(f"Notificação de conclusão enviada com SUCESSO para: {finish_url}")
        return {"status": "success"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Erro no upload/notificação: {e}"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Adicione seus argumentos aqui
    cli_args = parser.parse_args()
    execute_separation(cli_args)

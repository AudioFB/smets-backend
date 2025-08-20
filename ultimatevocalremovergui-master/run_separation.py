# run_separation.py
import os
import argparse
import wget
import uuid
from argparse import Namespace
import torch
# --- CORREÇÃO AQUI ---
from demucs.hdemucs import HTDemucs 

# --- Adicionar esta linha para resolver o erro do PyTorch ---
torch.serialization.add_safe_globals([HTDemucs])

# Importa as classes de separação que você já tem
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC
from lib_v5.vr_network.model_param_init import ModelParameters

# --- DEFINIÇÃO DE CONSTANTES (Muitas são do seu api.py original) ---
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

# --- Configuração dos diretórios ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
INPUT_FOLDER = os.path.join(BASE_DIR, 'inputs')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Separa faixas de áudio usando modelos UVR.')
    parser.add_argument('--model-name', required=True, help='Nome do modelo a ser usado.')
    parser.add_argument('--process-method', required=True, help='Método de processamento (e.g., MDX-Net, Demucs).')
    parser.add_argument('--audio-url', required=True, help='URL do arquivo de áudio para processar.')
    
    args = parser.parse_args()

    print(f"Iniciando processo para o modelo: {args.model_name}")
    print(f"Método: {args.process_method}")
    print(f"Baixando áudio de: {args.audio_url}")

    # --- Baixar o arquivo de áudio ---
    job_id = str(uuid.uuid4())
    input_filename = f"{job_id}.wav" 
    input_path = os.path.join(INPUT_FOLDER, input_filename)
    
    try:
        wget.download(args.audio_url, input_path)
        print("\nDownload do áudio concluído.")
    except Exception as e:
        print(f"\nErro ao baixar o áudio: {e}")
        return

    # --- Preparar os dados para o processo de separação ---
    process_data = {
        'audio_file': input_path,
        'audio_file_base': os.path.splitext(os.path.basename(input_path))[0],
        'export_path': OUTPUT_FOLDER,
        'set_progress_bar': lambda *args, **kwargs: None, 
        'write_to_console': lambda text, base_text="": print(text),
        'process_iteration': lambda: None,
        'cached_source_callback': lambda *args, **kwargs: (None, None),
        'cached_model_source_holder': lambda *args, **kwargs: None,
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False,
        'list_all_models': []
    }

    params = {}
    model_path = ""
    
    # --- Lógica para encontrar o caminho do modelo e definir parâmetros ---
    if args.process_method == MDX_ARCH_TYPE:
        model_filename_onnx = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{args.model_name}.onnx')
        model_filename_ckpt = os.path.join(MODELS_FOLDER, 'MDX_Net_Models', f'{args.model_name}.ckpt')

        if os.path.exists(model_filename_onnx):
            model_path = model_filename_onnx
            params['is_mdx_ckpt'] = False
        elif os.path.exists(model_filename_ckpt):
            model_path = model_filename_ckpt
            params['is_mdx_ckpt'] = True
        else:
            print(f"Modelo MDX-Net não encontrado: {args.model_name}")
            return
            
        params['primary_stem'] = VOCAL_STEM

    elif args.process_method == DEMUCS_ARCH_TYPE:
        model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', 'v3_v4_repo', f'{args.model_name}.yaml')
        if not os.path.exists(model_path):
             model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', f'{args.model_name}.ckpt')

        if '4s' in args.model_name:
            params['demucs_stem_count'], params['demucs_source_map'] = 4, DEMUCS_4_SOURCE_MAPPER
        else:
            params['demucs_stem_count'], params['demucs_source_map'] = 2, DEMUCS_2_SOURCE_MAPPER

    # --- Bloco completo de atributos padrão restaurado ---
    model_data = Namespace(
        **{
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
                process_method=args.process_method, model_path=model_path,
                model_name=args.model_name, model_basename=args.model_name
            )
        }
    )

    # --- Executar a separação ---
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
            print("\nSeparação concluída!")
            print(f"Arquivos de saída salvos em: {OUTPUT_FOLDER}")
            result_files = os.listdir(OUTPUT_FOLDER)
            print("Arquivos gerados:", result_files)
        else:
            print(f"Método de processamento não suportado: {model_data.process_method}")

    except Exception as e:
        print(f"\nOcorreu um erro durante a separação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

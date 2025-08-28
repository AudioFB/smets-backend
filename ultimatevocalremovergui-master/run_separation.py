# run_separation.py (versão FINAL, baseada no seu script original e corrigida para RunPod)
import os
import argparse
import requests
import uuid
import json
import hashlib
import boto3
from argparse import Namespace
import torch
from demucs.hdemucs import HDemucs as HTDemucs
import zipfile
import traceback

# --- Imports e Configurações Iniciais (do seu script original) ---
torch.serialization.add_safe_globals([HTDemucs])
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC

# --- DEFINIÇÃO DE CONSTANTES (do seu script original) ---
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

# --- Configuração dos diretórios (do seu script original) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# --- Funções Auxiliares (do seu script original) ---
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

def main():
    # --- ETAPA 1: Análise dos Argumentos (ADAPTADO PARA O RUNPOD) ---
    parser = argparse.ArgumentParser(description='Separa faixas de áudio usando modelos UVR.')
    parser.add_argument('--jobId', required=True, help='ID único do Job.')
    parser.add_argument('--filename', required=True, help='Nome do arquivo de áudio original.')
    parser.add_argument('--baseUrl', required=True, help='URL base do servidor web.')
    parser.add_argument('--isRunPod', default="False", help='Flag para indicar se está rodando no RunPod.')
    parser.add_argument('--model_name', required=True, help='Nome do modelo a ser usado.')
    parser.add_argument('--process_method', required=True, help='Método de processamento.')

    args = parser.parse_args()
    
    # Adiciona um print logo no início para vermos no log se o script começou
    print(f"--- run_separation.py iniciado para o Job ID: {args.jobId} ---")

    # --- ETAPA 2: Preparar Ambiente e Arquivo de Entrada (ADAPTADO PARA O RUNPOD) ---
    work_dir = f"/tmp/{args.jobId}"
    input_path = os.path.join(work_dir, args.filename)
    output_folder = work_dir # Usaremos o mesmo diretório para os arquivos de saída

    # --- ETAPA 3: Lógica de Processamento (DO SEU SCRIPT ORIGINAL, SEM MUDANÇAS) ---
    process_data = {
        'audio_file': input_path, 'audio_file_base': os.path.splitext(os.path.basename(input_path))[0],
        'export_path': output_folder, 
        'set_progress_bar': lambda *args, **kwargs: None, 
        'write_to_console': lambda text, base_text="": print(text), 'process_iteration': lambda: None,
        'cached_source_callback': lambda *args, **kwargs: (None, None),
        'cached_model_source_holder': lambda *args, **kwargs: None,
        'is_ensemble_master': False, 'is_4_stem_ensemble': False, 'list_all_models': []
    }

    params = {}
    model_path = ""
    
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
            print(f"Modelo MDX-Net não encontrado: {args.model_name}")
            return
        
        model_hash = get_model_hash(model_path)
        model_params_json = MDX_MODEL_PARAMS.get(model_hash, {})
        params['compensate'] = model_params_json.get('compensate', 1.035)
        params['mdx_dim_f_set'] = model_params_json.get('mdx_dim_f_set')
        params['mdx_dim_t_set'] = model_params_json.get('mdx_dim_t_set')
        params['mdx_n_fft_scale_set'] = model_params_json.get('mdx_n_fft_scale_set')
        params['primary_stem'] = model_params_json.get('primary_stem', VOCAL_STEM)

    elif args.process_method == DEMUCS_ARCH_TYPE:
        model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', 'v3_v4_repo', f'{args.model_name}.yaml')
        if not os.path.exists(model_path):
             model_path = os.path.join(MODELS_FOLDER, 'Demucs_Models', f'{args.model_name}.ckpt')

        if '6s' in args.model_name:
             params['demucs_stem_count'] = 6
        elif '4s' in args.model_name:
            params['demucs_stem_count'] = 4
        else:
            params['demucs_stem_count'] = 2

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
                demucs_source_map={}, demucs_stem_count=0, is_gpu_conversion=True, device_set='0',
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
        else:
            print(f"Método de processamento não suportado: {model_data.process_method}")
            return

    except Exception as e:
        print(f"\nOcorreu um erro durante a separação: {e}")
        traceback.print_exc()
        return

    # --- ETAPA 4: Compactar e Fazer Upload dos Resultados para o R2 ---
    try:
        # --- Compactação (sem alterações) ---
        zip_path = os.path.join(work_dir, f"{args.jobId}.zip")
        print(f"Criando arquivo zip em: {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac')) and file != args.filename:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))
        print("Compactação concluída.")

        # --- Novo Upload para o R2 ---
        
        # Pega as credenciais das variáveis de ambiente seguras do RunPod
        endpoint_url = os.environ.get('R2_ENDPOINT_URL')
        access_key_id = os.environ.get('R2_ACCESS_KEY_ID')
        secret_access_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        bucket_name = os.environ.get('R2_BUCKET_NAME')
        public_domain = os.environ.get('R2_PUBLIC_DOMAIN')
        
        # Configura o cliente para se conectar ao Cloudflare R2
        s3 = boto3.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto' # Região padrão para R2
        )
        
        zip_filename = f"{args.jobId}.zip"
        print(f"Enviando {zip_filename} para o bucket R2: {bucket_name}")
        
        # Faz o upload do arquivo para o bucket
        s3.upload_file(zip_path, bucket_name, zip_filename)
        
        # Constrói a URL pública final para o usuário baixar o arquivo
        download_url = f"https://{public_domain}/{zip_filename}"
        print(f"Upload para R2 concluído! URL de download: {download_url}")
        
        # --- Notifica seu servidor que o trabalho terminou ---
        finish_url = f'{args.baseUrl}/mixbuster/finish_job.php'
        payload = {'jobId': args.jobId, 'downloadUrl': download_url}
        
        # Adiciona os mesmos headers para passar pelo firewall
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*'
        }

        # Envia a notificação com os headers e verificação de erro
        response = requests.post(finish_url, data=payload, headers=headers)
        response.raise_for_status() # Garante que veremos um erro se a notificação falhar

        print(f"Notificação de conclusão enviada COM SUCESSO para: {finish_url}")


    except Exception as e:
        print(f"\nOcorreu um erro durante a compactação ou upload para o R2: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()











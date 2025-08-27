# run_separation.py (versão final compatível com o handler do RunPod)
import os
import argparse
import requests
import zipfile
import hashlib
import json
from argparse import Namespace
import torch
from demucs.hdemucs import HDemucs as HTDemucs

# --- Imports e Configurações Iniciais ---
torch.serialization.add_safe_globals([HTDemucs])
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC

# --- DEFINIÇÃO DE CONSTANTES ---
DEMUCS_ARCH_TYPE = 'Demucs'
MDX_ARCH_TYPE = 'MDX-Net'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
# ... (outras constantes podem ser adicionadas se necessário) ...

# --- Configuração dos diretórios base ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# --- Funções Auxiliares (mantidas do seu original) ---
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
    # --- ETAPA 1: Análise dos Argumentos (Corrigida para o RunPod) ---
    parser = argparse.ArgumentParser(description='Separa faixas de áudio usando modelos UVR.')
    parser.add_argument('--jobId', required=True, help='ID único do Job.')
    parser.add_argument('--filename', required=True, help='Nome do arquivo de áudio original.')
    parser.add_argument('--baseUrl', required=True, help='URL base do servidor web.')
    parser.add_argument('--isRunPod', default="False", help='Flag para indicar se está rodando no RunPod.')
    parser.add_argument('--model_name', required=True, help='Nome do modelo a ser usado.')
    parser.add_argument('--process_method', required=True, help='Método de processamento.')

    args = parser.parse_args()

    job_id = args.jobId
    filename = args.filename
    base_url = args.baseUrl
    is_runpod = args.isRunPod.lower() == 'true'

    print(f"Iniciando Job ID: {job_id}")
    print(f"Modelo: {args.model_name}, Método: {args.process_method}")

    # --- ETAPA 2: Preparar Ambiente e Arquivo de Entrada (Dinâmico) ---
    work_dir = f"/tmp/{job_id}"
    print(f"Ambiente RunPod. Usando diretório de trabalho: {work_dir}")
    input_path = os.path.join(work_dir, filename)
    output_folder = work_dir

    # --- ETAPA 3: Lógica de Processamento (Seu código original, adaptado) ---
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
    
    # ... (Sua lógica original para encontrar o caminho do modelo e definir os parâmetros vai aqui) ...
    # ... (Esta parte parece correta no seu script original e pode ser mantida) ...
    if args.process_method == MDX_ARCH_TYPE:
        # ... (sua lógica mdx) ...
    elif args.process_method == DEMUCS_ARCH_TYPE:
        # ... (sua lógica demucs) ...

    model_data = Namespace(**{**dict(...), **params, **dict(...)}) # Sua lógica original de Namespace

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
            print(f"Método de processamento não suportado: {model_data.process_method}")
            return
    except Exception as e:
        print(f"\nOcorreu um erro durante a separação: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- ETAPA 4: Compactar e Fazer Upload dos Resultados ---
    try:
        zip_path = os.path.join(work_dir, f"{job_id}.zip")
        print(f"Criando arquivo zip em: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac')):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))
        
        print("Compactação concluída.")
        
        upload_url = f'{base_url}/mixbuster/upload_result.php'
        print(f"Enviando resultado para: {upload_url}")
        
        with open(zip_path, 'rb') as f:
            files = {'file': (f"{job_id}.zip", f)}
            data = {'jobId': job_id}
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
        
        print(f"Upload finalizado. Resposta do servidor: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"\nOcorreu um erro durante a compactação ou upload: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# runpod_handler.py
import os
import requests
import runpod
from argparse import Namespace
import sys
import torch
from pathlib import Path

# Adiciona o diretório do script de separação ao path do Python
sys.path.append('ultimatevocalremovergui-master')
from run_separation import execute_separation

# --- LÓGICA DE DOWNLOAD DE MODELOS ---
BASE_MODEL_URL = "https://github.com/AudioFB/smets-backend/releases/download/v1.0.0-models/"

# Mapeamento de nomes de modelos para seus arquivos e destinos
MODEL_FILES = {
    "htdemucs": {
        "files": ["htdemucs.yaml", "955717e8-8726e21a.th"],
        "dest": "ultimatevocalremovergui-master/models/Demucs_Models/v3_v4_repo"
    },
    "htdemucs_6s": {
        "files": ["htdemucs_6s.yaml", "5c90dfd2-34c22ccb.th"],
        "dest": "ultimatevocalremovergui-master/models/Demucs_Models/v3_v4_repo"
    },
    "Reverb_HQ_By_FoxJoy": {
        "files": ["Reverb_HQ_By_FoxJoy.onnx"],
        "dest": "ultimatevocalremovergui-master/models/MDX_Net_Models"
    }
}

def download_model_if_needed(model_name):
    """
    Verifica se os arquivos de um modelo existem localmente.
    Se não existirem, faz o download deles.
    """
    if model_name not in MODEL_FILES:
        print(f"Aviso: Modelo '{model_name}' não está mapeado para download.")
        return

    model_info = MODEL_FILES[model_name]
    dest_path = Path(model_info["dest"])
    dest_path.mkdir(parents=True, exist_ok=True) # Cria o diretório de destino se não existir

    for filename in model_info["files"]:
        file_path = dest_path / filename
        if file_path.exists():
            print(f"Modelo '{filename}' já existe localmente. Pulando download.")
            continue

        url = BASE_MODEL_URL + filename
        print(f"Baixando modelo '{filename}' de {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download de '{filename}' concluído.")
        except requests.exceptions.RequestException as e:
            # Se o download falhar, o job deve parar.
            raise RuntimeError(f"Falha ao baixar o modelo '{filename}': {e}") from e

def handler(job):
    # Bloco de verificação da GPU...
    print("--- VERIFICAÇÃO DE AMBIENTE SERVERLESS ---")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA está disponível? {is_cuda_available}")
    if is_cuda_available:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO CRÍTICO: GPU não encontrada.")
    print("-----------------------------------------")

    job_input = job['input']
    
    try:
        args = Namespace(
            jobId=job_input['jobId'],
            audioUrl=job_input['audioUrl'],
            filename=job_input['originalFilename'],
            model_name=job_input['model_name'],
            process_method=job_input['process_method'],
            baseUrl=job_input['baseUrl'],
            isRunPod="True"
        )
    except KeyError as e:
        return {"error": f"Parâmetro obrigatório ausente no input: {e}"}

    # --- ETAPA DE OTIMIZAÇÃO: BAIXAR APENAS O MODELO NECESSÁRIO ---
    try:
        download_model_if_needed(args.model_name)
    except RuntimeError as e:
        return {"error": str(e)}

    # O resto do seu código continua normalmente...
    work_dir = f"/tmp/{args.jobId}"
    os.makedirs(work_dir, exist_ok=True)
    input_path = os.path.join(work_dir, args.filename)

    try:
        print(f"Baixando arquivo de áudio de: {args.audioUrl}")
        response = requests.get(args.audioUrl, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        response.raise_for_status()
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download do áudio concluído.")
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar o arquivo do Cloudflare R2: {e}"}

    result = execute_separation(args)

    if result.get("error"):
        return result

    print("Handler concluído com sucesso.")
    return {"status": "success", "jobId": args.jobId}

runpod.serverless.start({"handler": handler})


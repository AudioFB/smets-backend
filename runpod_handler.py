# runpod_handler.py
import os
import requests
import runpod
from argparse import Namespace
import sys
import torch  # <-- 1. IMPORTAMOS O TORCH AQUI

# Adiciona o diretório do script de separação ao path do Python
sys.path.append('ultimatevocalremovergui-master')
from run_separation import execute_separation

def handler(job):
    # --- 2. ADICIONAMOS O BLOCO DE VERIFICAÇÃO DA GPU AQUI ---
    print("--- VERIFICAÇÃO DE AMBIENTE SERVERLESS ---")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA está disponível? {is_cuda_available}")

    if is_cuda_available:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"Versão PyTorch: {torch.__version__}")
    else:
        print("AVISO CRÍTICO: A GPU não foi encontrada. O processo rodará em CPU.")
    
    print("-----------------------------------------")
    # --- FIM DA VERIFICAÇÃO ---

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

    work_dir = f"/tmp/{args.jobId}"
    os.makedirs(work_dir, exist_ok=True)
    input_path = os.path.join(work_dir, args.filename)

    try:
        print(f"Baixando arquivo de: {args.audioUrl}")
        response = requests.get(args.audioUrl, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        response.raise_for_status()
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar o arquivo do Cloudflare R2: {e}"}

    result = execute_separation(args)

    if result.get("error"):
        return result

    print("Handler concluído com sucesso.")
    return {"status": "success", "jobId": args.jobId}

runpod.serverless.start({"handler": handler})

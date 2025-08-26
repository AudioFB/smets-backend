# runpod_handler.py
import subprocess
import os
import requests
import zipfile
import runpod

# Esta é a função que o RunPod irá chamar automaticamente.
def handler(job):
    job_input = job['input']
    
    # Validação para garantir que recebemos os dados do PHP
    try:
        job_id = job_input['jobId']
        filename = job_input['filename']
        base_url = job_input['baseUrl']
    except KeyError as e:
        return {"error": f"Faltando parâmetro obrigatório: {e}"}

    # --- Etapa 1: Preparar o ambiente dentro do container ---
    
    # Diretório de trabalho temporário para este job específico
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    # Monta a URL completa para baixar o áudio do seu servidor
    download_url = f"{base_url}/mixbuster/uploads/{job_id}/{filename}"
    input_path = os.path.join(work_dir, filename)

    # --- Etapa 2: Baixar o arquivo de áudio ---
    
    try:
        print(f"Baixando arquivo de: {download_url}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Verifica se o download foi bem-sucedido (código 200)
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar o arquivo: {e}"}

    # --- Etapa 3: Executar o script de separação ---
    
    # O comando é o mesmo que você usa no GitHub Actions.
    # Note que agora passamos a baseUrl para o script.
    script_path = "ultimatevocalremovergui-master/run_separation.py"
    command = [
        "python", script_path,
        "--jobId", job_id,
        "--filename", filename,
        "--baseUrl", base_url, # Passamos a URL base para o script
        "--isRunPod", "True"   # Flag para o script saber que está no RunPod
    ]
    
    print(f"Executando comando: {' '.join(command)}")
    
    # Executa o processo e captura a saída para depuração
    process = subprocess.run(command, capture_output=True, text=True, cwd=".")
    
    # Se o script falhar, retornamos o erro detalhado
    if process.returncode != 0:
        print("--- ERRO NO SCRIPT DE SEPARAÇÃO ---")
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        print("------------------------------------")
        return {
            "error": "Erro durante a execução do script de separação.",
            "stdout": process.stdout,
            "stderr": process.stderr
        }

    # --- Etapa 4: Retornar sucesso ---
    
    # O próprio run_separation.py fará o upload do .zip.
    # Aqui, apenas confirmamos que o handler terminou com sucesso.
    print("Processo concluído com sucesso.")
    return {"status": "success", "jobId": job_id}

# Inicia o handler para o RunPod
runpod.serverless.start({"handler": handler})
# runpod_handler.py (versão final com underscores)
import subprocess
import os
import requests
import runpod

def handler(job):
    job_input = job['input']
    
    try:
        job_id = job_input['jobId']
        filename = job_input['filename']
        base_url = job_input['baseUrl']
        # *** CORREÇÃO AQUI: Lendo as chaves com underscore ***
        model_name = job_input['model_name']
        process_method = job_input['process_method']
    except KeyError as e:
        return {"error": f"Faltando parâmetro obrigatório no input: {e}"}

    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    download_url = f"{base_url}/mixbuster/uploads/{job_id}/{filename}"
    input_path = os.path.join(work_dir, filename)

    try:
        print(f"Baixando arquivo de: {download_url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar o arquivo: {e}"}

    script_path = "ultimatevocalremovergui-master/run_separation.py"
    
    # *** CORREÇÃO AQUI: Usando underscores nos argumentos de linha de comando ***
    command = [
        "python3.13", script_path,
        "--jobId", job_id,
        "--filename", filename,
        "--baseUrl", base_url,
        "--isRunPod", "True",
        "--model_name", model_name,
        "--process_method", process_method
    ]
    
    print(f"Executando comando: {' '.join(command)}")
    
    process = subprocess.run(command, capture_output=True, text=True, cwd=".")
    
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

    print("Processo concluído com sucesso.")
    return {"status": "success", "jobId": job_id}

runpod.serverless.start({"handler": handler})



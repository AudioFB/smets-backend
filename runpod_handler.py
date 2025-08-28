# runpod_handler.py
import subprocess
import os
import requests
import runpod

def handler(job):
    job_input = job['input']
    
    try:
        job_id = job_input['jobId']
        # --- MUDANÇA AQUI ---
        audio_url = job_input['audioUrl'] # Recebe a URL completa do Cloudflare R2
        original_filename = job_input['originalFilename']
        model_name = job_input['model_name']
        process_method = job_input['process_method']
    except KeyError as e:
        return {"error": f"Faltando parâmetro obrigatório no input: {e}"}

    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    input_path = os.path.join(work_dir, original_filename)

    try:
        print(f"Baixando arquivo de: {audio_url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(audio_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar o arquivo do Cloudflare R2: {e}"}

    script_path = "ultimatevocalremovergui-master/run_separation.py"
    
    # O comando agora passa o nome do arquivo original, não precisa mais do baseUrl
    command = [
        "python3", script_path,
        "--jobId", job_id,
        "--filename", original_filename,
        "--model_name", model_name,
        "--process_method", process_method,
        # O baseUrl não é mais necessário aqui, mas o script de separação precisa ser ajustado
        # para não depender dele. Vamos garantir que `run_separation.py` também esteja correto.
        "--baseUrl", "https://cubesoundlab.com/dev/letsdaw" # Manter por enquanto para compatibilidade
    ]
    
    print(f"Executando comando: {' '.join(command)}")
    
    process = subprocess.run(command, capture_output=True, text=True, cwd=".")

    if process.returncode != 0:
        return {
            "error": "Erro durante a execução do script de separação.",
            "stdout": process.stdout,
            "stderr": process.stderr
        }

    return {"status": "success", "jobId": job_id}

runpod.serverless.start({"handler": handler})

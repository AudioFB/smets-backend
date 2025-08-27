#!/bin/bash
set -e

echo "--- [1/4] Atualizando pacotes do sistema ---"
apt-get update && apt-get install -y ffmpeg

echo "--- [2/4] Instalando dependências Python ---"
pip install -r requirements.txt

echo "--- [3/4] Verificando instalação do PyTorch ---"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "--- [4/4] Iniciando o handler do RunPod ---"
python3 -u runpod_handler.py

#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "FALHA na inicialização, linha $LINENO (exit $?)" >&2' ERR
cd "$(dirname "$0")"
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
echo '--- ETAPA 1: PACOTES DO SISTEMA ---'
apt-get update
apt-get install -y --no-install-recommends python3.11-venv python3.11-tk cmake build-essential libsndfile1 ffmpeg libgl1 libglib2.0-0
# Ambiente isolado: não herda os pacotes incompatíveis do container anterior.
echo '--- ETAPA 2: AMBIENTE PYTHON ISOLADO ---'
python3.11 -m venv .venv
PY="$PWD/.venv/bin/python"
"$PY" -m pip install 'pip==25.1.1' 'setuptools==80.9.0' 'wheel==0.45.1'
echo '--- ETAPA 3: PYTORCH CUDA 12.8 ---'
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo '--- ETAPA 4: DEPENDÊNCIAS COMPATÍVEIS ---'
"$PY" -m pip install -r ultimatevocalremovergui-master/requirements.txt
"$PY" -m pip check
"$PY" -m pip freeze > installed-versions.txt
echo '--- ETAPA 5: IMPORTAÇÃO COMPLETA E GPU ---'
"$PY" preflight.py
echo '--- ETAPA 6: INICIANDO WORKER ---'
exec "$PY" -u runpod_handler.py

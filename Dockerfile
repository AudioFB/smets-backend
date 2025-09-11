# 1. Comece com a mesma imagem base que você usa no RunPod
FROM runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2004

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o arquivo de requisitos primeiro.
#    Isso otimiza o cache do Docker. Se você só mudar seu código,
#    esta etapa de instalação não precisará rodar de novo.
COPY ./app/ultimatevocalremovergui-master/requirements.txt .

# 4. Instale o pip e todas as dependências do Python
#    Tudo isso ficará "pré-instalado" na sua imagem.
RUN apt-get update && apt-get install -y curl python3.13-tk && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.13 get-pip.py && \
    python3.13 -m pip install --no-cache-dir -r requirements.txt && \
    python3.13 -m pip install --no-cache-dir --upgrade --force-reinstall runpod && \
    # Limpa o cache do apt para manter a imagem pequena
    rm -rf /var/lib/apt/lists/*

# 5. Copie todo o resto do seu código para dentro do container
COPY ./app /app

# 6. O comando para iniciar o worker (opcional, pode ser definido no RunPod)
CMD ["python3.13", "runpod_handler.py"]
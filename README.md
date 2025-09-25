# LetsDAW - Digital Audio Workstation Baseada na Web

Bem-vindo ao LetsDAW, uma Digital Audio Workstation (DAW) completa que roda diretamente no seu navegador, com um poderoso separador de stems por IA integrado, o **MixBuster**.

## Sobre o Projeto

LetsDAW foi projetado para fornecer uma experiência de produção de áudio acessível e poderosa sem a necessidade de instalações de software complexas. Ele permite que os usuários carreguem até 16 faixas de áudio, apliquem efeitos em tempo real (EQ, Compressor), utilizem plugins VSM customizados, gravem novas faixas e exportem o trabalho finalizado.

O coração do processamento de áudio é o **Csound**, uma robusta linguagem de programação de som, executada via WebAssembly para garantir performance e compatibilidade entre navegadores.

### Principais Funcionalidades

* **Mixer de 16 Canais:** Carregue e mixe até 16 faixas de áudio.
* **Processamento em Tempo Real:** Ajuste de volume, pan, mute, solo e efeitos aplicados instantaneamente.
* **Efeitos por Canal:** Cada faixa possui um Equalizador paramétrico de 3 bandas, Compressor e 3 slots para plugins de inserção.
* **Efeitos Globais (Aux Sends):** Efeitos de Reverb e Delay globais com controles de envio por canal.
* **Suporte a Plugins VSM:** Carregue plugins de áudio customizados no formato VSM tanto em canais individuais quanto no master.
* **Gravação Multipista:** Grave novas faixas a partir de uma interface de áudio, com seleção de canal de entrada e compensação de latência.
* **Gerenciamento de Sessão:** Exporte e importe sessões de mixagem completas (arquivos `.mix`), salvando áudios, parâmetros e plugins.
* **Exportação de Áudio:** Renderize a mixagem final nos formatos WAV ou MP3.
* **MixBuster (IA Stem Separation):** Separe uma faixa de áudio em seus componentes (vocais, bateria, baixo, etc.) usando modelos de IA e importe-os diretamente para a DAW.

### Stack de Tecnologia

* **Frontend:** HTML5, CSS3, JavaScript (Módulos ES6)
* **Motor de Áudio:** Csound (via WebAssembly com `@csound/browser`)
* **Backend:** PHP
* **Processamento IA (MixBuster):** RunPod (Servidor GPU sob demanda)
* **Armazenamento de Arquivos (MixBuster):** Cloudflare R2
* **Visualização de Áudio:** WaveSurfer.js

## Fluxo de Trabalho do MixBuster (Separação de Stems)

O MixBuster é um componente crucial que integra serviços externos para realizar a separação de áudio. O fluxo ocorre da seguinte forma:

1.  **Upload do Usuário:** O usuário seleciona um arquivo de áudio e um método de separação na interface do LetsDAW.
2.  **URL de Upload Segura:** O frontend (`mixbuster-integration.js`) solicita ao backend (`generate_upload_url.php`) uma URL pré-assinada (presigned URL) para fazer o upload seguro para o bucket do Cloudflare R2.
3.  **Upload para R2:** O arquivo de áudio é enviado diretamente do navegador do usuário para o Cloudflare R2, sem passar pelo servidor da aplicação.
4.  **Início do Job:** O frontend notifica o backend (`start_job.php`) que o upload foi concluído. O backend, por sua vez, faz uma chamada de API para o **RunPod**, enviando a URL pública do arquivo no R2 e os parâmetros do trabalho (como o modelo de IA a ser usado).
5.  **Processamento na GPU:** O worker no RunPod baixa o áudio, processa a separação dos stems e compacta os resultados em um arquivo `.zip`.
6.  **Notificação de Conclusão:** Ao terminar, o worker do RunPod envia o resultado (a nova URL pública do arquivo .zip no R2) de volta para o backend da aplicação (`finish_job.php`). O backend então cria um arquivo `status.json` local para sinalizar a conclusão.
7.  **Polling do Frontend:** Enquanto o processo ocorre, o frontend (`mixbuster-integration.js`) verifica periodicamente o status do job através do script `check_status.php`.
8.  **Resultados:** Assim que o `status.json` é encontrado, o frontend exibe os botões "Download Stems" e "Import to LetsDAW", permitindo que o usuário baixe o .zip ou importe as faixas de áudio diretamente para a DAW.

---

## Configuração de Ambiente

**IMPORTANTE:** Este projeto contém caminhos de arquivos e URLs que são específicos para o ambiente de desenvolvimento/homologação (`/stage/`). Ao mover o projeto para um ambiente de produção ou um diretório diferente, você **PRECISARÁ** atualizar os seguintes arquivos:

### 1. `letsdaw/mixbuster/config.php`

Este é o arquivo de configuração principal para o MixBuster. A constante `APP_BASE_URL` precisa ser alterada para refletir o novo domínio e caminho.

* **Linha a ser alterada:**
    ```php
    define('APP_BASE_URL', '[https://cubesoundlab.com/stage/letsdaw](https://cubesoundlab.com/stage/letsdaw)');
    ```

### 2. `letsdaw/js/mixbuster-integration.js`

Este arquivo contém as chamadas `fetch` do frontend para o backend. Os caminhos estão relativos à raiz do domínio, então eles incluem `/stage/`.

* **Linhas a serem alteradas (Exemplos):**
    ```javascript
    // Dentro da função handleProcessClick
    const presignResponse = await fetch(`/stage/letsdaw/mixbuster/generate_upload_url.php?filename=${encodeURIComponent(file.name)}`);
    const startJobResponse = await fetch('/stage/letsdaw/mixbuster/start_job.php', { ... });

    // Dentro da função pollForResults
    const response = await fetch(`/stage/letsdaw/mixbuster/check_status.php?jobId=${jobId}`);
    
    // Dentro da função displayResults
    fetch(`/stage/letsdaw/mixbuster/cleanup_job.php?jobId=${jobId}`);
    ```
    **Recomendação:** Para facilitar futuras migrações, considere criar uma variável global de `basePath` no seu `index.php` e usá-la para construir essas URLs dinamicamente no JavaScript.

### 3. (Opcional) `letsdaw/index.php`

No final deste arquivo, há uma lógica de `fetch` para debitar os "cubes" do usuário. A URL é construída dinamicamente com base na função `url_info()`, o que é uma boa prática. No entanto, é importante garantir que as variáveis de servidor (`$_SERVER`) estejam corretas no novo ambiente.

* **Linha a ser verificada:**
    ```php
    const response = await fetch('<?php echo "{$caminhos['protocolo']}://{$caminhos['host']}{$caminhos['diretorio_raiz']}/assets/php/cubeDebitaCubes.php"; ?>', { ... });
    ```
    Normalmente, isso deve funcionar sem alterações, mas é bom estar ciente.

---

1.  **Dependências do Backend:** O projeto utiliza o AWS SDK para PHP para interagir com o Cloudflare R2. Certifique-se de que o Composer está instalado e execute:
    ```bash
    composer install
    ```
    Isso criará o diretório `vendor/` e o arquivo `autoload.php`, que são necessários para os scripts do backend.
2.  **Configuração de Chaves:** Preencha todas as chaves e segredos nos arquivos de configuração, como `letsdaw/mixbuster/config.php`, com suas credenciais do RunPod e Cloudflare R2.
3.  **Permissões de Diretório:** Certifique-se de que o servidor web (Apache, Nginx) tenha permissão de escrita no diretório `letsdaw/mixbuster/uploads/`.
4.  **Configuração do CORS:** Ajuste a política de CORS no seu bucket do Cloudflare R2 para permitir requisições GET do seu domínio de produção.

<?php
// /mixbuster/check_status.php

require __DIR__ . '/../../vendor/autoload.php';
require 'config.php';

use Aws\S3\S3Client;
use Aws\Exception\S3Exception;

header('Content-Type: application/json');

$jobId = $_GET['jobId'] ?? null;
if (!$jobId) {
    http_response_code(400);
    echo json_encode(['status' => 'error', 'message' => 'Job ID não fornecido.']);
    exit;
}

$safeJobId = preg_replace('/[^a-zA-Z0-9-]/', '', $jobId);
$status_key = "{$safeJobId}-complete.json";
$zip_key = "{$safeJobId}-mixbusted.zip";

$s3Client = new S3Client([
    'region' => 'auto',
    'version' => 'latest',
    'endpoint' => "https://" . R2_ACCOUNT_ID . ".r2.cloudflarestorage.com",
    'credentials' => [
        'key' => R2_ACCESS_KEY_ID,
        'secret' => R2_SECRET_ACCESS_KEY,
    ]
]);

try {
    // Tenta obter o objeto de status do R2
    $result = $s3Client->getObject([
        'Bucket' => R2_BUCKET_NAME,
        'Key' => $status_key,
    ]);

    // Se chegou aqui, o arquivo existe. O trabalho está concluído.
    $status_data = $result['Body']->getContents();
    
    // Deleta os arquivos do R2 para limpeza automática
    $s3Client->deleteObject(['Bucket' => R2_BUCKET_NAME, 'Key' => $status_key]);
    $s3Client->deleteObject(['Bucket' => R2_BUCKET_NAME, 'Key' => $zip_key]);
    
    // Deleta o diretório local de status, se existir
    $localJobDir = __DIR__ . '/uploads/' . $safeJobId;
    if (is_dir($localJobDir)) {
        rmdir($localJobDir);
    }
    
    // Retorna os dados de conclusão para o frontend
    echo $status_data;

} catch (S3Exception $e) {
    // A exceção mais comum será 'NoSuchKey', o que significa que o trabalho ainda não terminou.
    if ($e->getAwsErrorCode() === 'NoSuchKey') {
        echo json_encode(['status' => 'pending']);
    } else {
        // Outro erro de S3, retorna o erro
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => 'Erro ao verificar status no R2.', 'details' => $e->getMessage()]);
    }
}
?>

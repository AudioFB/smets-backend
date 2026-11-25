"""Falha antes de iniciar o worker se imports ou execução ONNX/CUDA falharem."""
import torch  # Carrega as bibliotecas CUDA/cuDNN antes do ONNX Runtime.
import numpy as np
import onnx
import onnxruntime as ort
import cv2
import pandas
import ml_dtypes
from onnx import helper, TensorProto

print('Versões:', {'torch': torch.__version__, 'numpy': np.__version__,
                  'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
                  'opencv': cv2.__version__, 'pandas': pandas.__version__}, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('GPU CUDA não disponível.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
graph = helper.make_graph(
    [helper.make_node('MatMul', ['x', 'y'], ['z'])], 'cuda_smoke',
    [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 2]) for name in ('x', 'y')],
    [helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
session = ort.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider'])
if 'CUDAExecutionProvider' not in session.get_providers():
    raise RuntimeError('ONNX não conseguiu inicializar CUDAExecutionProvider.')
values = np.ones((2, 2), dtype=np.float32)
np.testing.assert_allclose(session.run(None, {'x': values, 'y': values})[0], values * 2)
import runpod_handler  # Inclui separate, Demucs e todas as dependências do projeto.
print('PREFLIGHT OK: importação completa e sessão ONNX/CUDA passaram.', flush=True)

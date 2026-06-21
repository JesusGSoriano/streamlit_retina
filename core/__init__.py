"""Núcleo del pipeline de análisis de imágenes retinianas (TFM / CIPF).

Lógica separada de la interfaz: estos módulos reciben imágenes/arrays y
devuelven resultados, sin visualización. Reutilizables desde Streamlit o desde
una app de escritorio.
"""

from .device import DEVICE, USE_GPU, get_device
from .preprocessing import preprocess_retinal_image, create_fov_mask
from .segmentation import segment_vessels, frangi_multiscale_gpu
from .metrics import compute_vascular_area, compute_skeleton_metrics
from .leakage import detect_fluorescein_leakage
from .classifier import load_ensemble, classify_image, RetinaEnsemble
from .pipeline import analyze_image

__all__ = [
    'DEVICE', 'USE_GPU', 'get_device',
    'preprocess_retinal_image', 'create_fov_mask',
    'segment_vessels', 'frangi_multiscale_gpu',
    'compute_vascular_area', 'compute_skeleton_metrics',
    'detect_fluorescein_leakage',
    'load_ensemble', 'classify_image', 'RetinaEnsemble',
    'analyze_image',
]

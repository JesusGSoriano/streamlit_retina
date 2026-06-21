"""Detección automática del dispositivo de cómputo (GPU si está disponible, CPU si no).

En el despliegue (p.ej. Streamlit Community Cloud) será CPU; en local con CUDA
usará la GPU sin cambiar nada del resto del código.
"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()


def get_device() -> torch.device:
    return DEVICE

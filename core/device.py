"""Dispositivo de cómputo para la demo de Streamlit.

En Streamlit Cloud el despliegue es siempre CPU, así que aquí fijamos CPU y no
hay detección ni uso de GPU. La aceleración por GPU se reserva para la futura
app de escritorio, que reutilizará estos módulos con su propia configuración.
"""

import torch

DEVICE = torch.device('cpu')


def get_device() -> torch.device:
    return DEVICE

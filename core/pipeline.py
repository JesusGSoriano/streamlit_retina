"""Orquestador del pipeline de análisis (sin visualización).

Reproducimos la función `analyze_image` del notebook: preprocesamos, segmentamos,
extraemos las métricas vasculares y detectamos las fugas. Devolvemos los arrays y
las métricas en un dict, para que la interfaz (Streamlit, escritorio...) se limite
a mostrarlos.

Dejamos la clasificación aparte (core/classifier.py) porque el ensemble lo
cargamos una sola vez y lo reutilizamos.
"""

import os

import numpy as np
import torch

from .device import DEVICE
from .preprocessing import preprocess_retinal_image
from .segmentation import segment_vessels
from .metrics import compute_vascular_area, compute_skeleton_metrics
from .leakage import detect_fluorescein_leakage

# Métricas escalares que exponemos (sin arrays internos como diámetros/esqueleto)
_SKEL_SCALAR_KEYS = (
    'n_vessel_segments', 'calibre_px', 'std_diameter_px',
    'mean_tortuosity', 'std_tortuosity', 'total_length_px',
)


def analyze_image(path: str, device: torch.device = None) -> dict:
    """Ejecutamos el pipeline completo sobre una imagen.

    Devolvemos un dict con:
      - arrays: img_rgb, img_green, img_clahe, fov_mask, vessel_mask,
                frangi_response, leakage_mask, leak_props, skeleton, dist_transform
      - métricas: vessel_density_pct, n_vessel_segments, calibre_px,
                  mean_tortuosity, total_length_px, n_leaks, total_leak_area_px, ...
      - error: None si todo fue bien, o un mensaje.
    """
    if device is None:
        device = DEVICE

    result = {
        'path': path,
        'filename': os.path.basename(path),
        'error': None,
        'img_rgb': None, 'img_green': None, 'img_clahe': None, 'fov_mask': None,
        'vessel_mask': None, 'frangi_response': None,
        'leakage_mask': None, 'leak_props': [],
        'skeleton': None, 'dist_transform': None,
        'vessel_diameters': [], 'vessel_tortuosities': [],
    }

    try:
        img_rgb, img_green, img_clahe, fov_mask = preprocess_retinal_image(path)

        if fov_mask.sum() < 1000:
            result['error'] = 'FOV mask too small'
            return result

        vessel_mask, frangi_resp = segment_vessels(img_clahe, fov_mask, device=device)

        result.update(compute_vascular_area(vessel_mask, fov_mask))

        skel_metrics = compute_skeleton_metrics(vessel_mask)
        for k in _SKEL_SCALAR_KEYS:
            result[k] = skel_metrics.get(k)
        result['vessel_diameters'] = skel_metrics.get('vessel_diameters', [])
        result['vessel_tortuosities'] = skel_metrics.get('vessel_tortuosities', [])
        result['skeleton'] = skel_metrics.get('skeleton')
        result['dist_transform'] = skel_metrics.get('dist_transform')

        leak_metrics = detect_fluorescein_leakage(
            img_clahe, fov_mask, vessel_mask=vessel_mask, frangi_response=frangi_resp
        )
        result['max_leak_area_px'] = max(
            (p.area for p in leak_metrics['leak_props']), default=0
        )
        for k, v in leak_metrics.items():
            if k != 'leak_props':
                result[k] = v
        result['leak_props'] = leak_metrics['leak_props']

        result['img_rgb'] = img_rgb
        result['img_green'] = img_green
        result['img_clahe'] = img_clahe
        result['fov_mask'] = fov_mask
        result['vessel_mask'] = vessel_mask
        result['frangi_response'] = frangi_resp
        result['img_height'] = img_rgb.shape[0]
        result['img_width'] = img_rgb.shape[1]

    except Exception as e:
        result['error'] = str(e)

    return result

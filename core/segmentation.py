"""Segmentación de la red vascular.

Filtro de Frangi multiescala implementado en PyTorch (funciona en GPU o CPU)
combinado con top-hat morfológico. Portado literalmente del notebook del TFM.
"""

import numpy as np
import torch
import torch.nn.functional as F

from scipy import ndimage
from skimage import morphology
from skimage.morphology import disk

from .device import DEVICE, USE_GPU
from .preprocessing import tophat_gpu


def frangi_multiscale_gpu(img_clahe: np.ndarray,
                          fov_mask: np.ndarray,
                          scales: list = None,
                          device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = DEVICE
    if scales is None:
        # Escalas más finas y con más resolución en los vasos pequeños.
        # Empezamos en 0.5 para capturar capilares finos que con sigma=1 se perdían.
        scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

    H, W = img_clahe.shape

    # Imagen invertida (vasos brillantes → oscuros para Frangi)
    img_inv = 1.0 - img_clahe
    t = torch.tensor(img_inv, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    responses = []

    for sigma in scales:
        ks = int(6 * sigma + 1) | 1
        pad = ks // 2
        ax = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2

        g = torch.exp(-ax**2 / (2 * sigma**2))
        g = g / g.sum()
        dg = -ax / (sigma**2) * g                    # Primera derivada
        d2g = (ax**2 / sigma**4 - 1/sigma**2) * g     # Segunda derivada

        kxx = (d2g[:, None] * g[None, :]).view(1, 1, ks, ks)
        kyy = (g[:, None] * d2g[None, :]).view(1, 1, ks, ks)
        kxy = (dg[:, None] * dg[None, :]).view(1, 1, ks, ks)

        Hxx = F.conv2d(t, kxx, padding=pad).squeeze() * sigma**2
        Hyy = F.conv2d(t, kyy, padding=pad).squeeze() * sigma**2
        Hxy = F.conv2d(t, kxy, padding=pad).squeeze() * sigma**2

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy**2
        disc = torch.sqrt(torch.clamp((trace**2 - 4*det), min=0))
        l1 = 0.5 * (trace + disc)
        l2 = 0.5 * (trace - disc)

        swap = torch.abs(l1) > torch.abs(l2)
        lam1 = torch.where(swap, l2, l1)
        lam2 = torch.where(swap, l1, l2)

        beta = 0.5
        Rb = (lam1 / (lam2 + 1e-8))**2
        S2 = lam1**2 + lam2**2

        # gamma adaptativo: la mitad del máximo de S por escala, en vez de fijo.
        # Esto hace que el filtro se ajuste al contraste real de cada imagen,
        # detectando vasos tenues que con gamma=15 fijo se quedaban fuera.
        gamma2 = 0.5 * S2.max()
        gamma2 = torch.clamp(gamma2, min=1e-6)

        valid = lam2 < 0
        vesselness = (
            torch.exp(-Rb / (2 * beta**2)) *
            (1 - torch.exp(-S2 / (2 * gamma2)))
        )
        vesselness = torch.where(valid, vesselness, torch.zeros_like(vesselness))
        responses.append(vesselness)

    frangi_resp = torch.stack(responses, dim=0).max(dim=0).values

    fmax = frangi_resp.max()
    if fmax > 0:
        frangi_resp = frangi_resp / fmax

    return frangi_resp


def segment_vessels_gpu(img_clahe: np.ndarray, fov_mask: np.ndarray,
                        device: torch.device = None,
                        border_margin: int = 15,
                        seed_pct: float = 0.90,
                        expand_pct: float = 0.62,
                        min_object_size: int = 50) -> tuple:
    if device is None:
        device = DEVICE
    H, W = img_clahe.shape
    fov_t = torch.tensor(fov_mask.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)
    img_t = torch.tensor(img_clahe.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)

    frangi_resp = frangi_multiscale_gpu(img_clahe, fov_mask, device=device)
    frangi_resp = frangi_resp * fov_t.squeeze()

    tophat_resp = tophat_gpu(img_t, kernel_size=25).squeeze()
    tophat_resp = tophat_resp * fov_t.squeeze()
    th_max = tophat_resp.max()
    if th_max > 0:
        tophat_resp = tophat_resp / th_max

    combined = 0.7 * frangi_resp + 0.3 * tophat_resp
    combined = combined * fov_t.squeeze()

    # Excluimos un margen del borde del FOV. El anillo brillante del borde de la
    # imagen (arriba/abajo/izquierda/derecha del círculo del ojo) se colaba como
    # vaso y falseaba las métricas. Erosionamos el FOV y trabajamos dentro de esa
    # región tanto para calcular los umbrales (para que el brillo del borde no los
    # sesgue) como para la máscara final.
    fov_inner = morphology.binary_erosion(fov_mask.astype(bool), disk(border_margin))
    fov_inner_t = torch.tensor(fov_inner.astype(np.float32), device=device).bool()

    fov_vals = combined[fov_inner_t]

    # Histéresis más restrictiva (semillas p90, expansión p62) para ceñirse al
    # trazado fino real del vaso y no ensanchar la máscara.
    thresh_hi = torch.quantile(fov_vals, seed_pct)
    thresh_lo = torch.quantile(fov_vals, expand_pct)

    seeds_np = ((combined > thresh_hi) & fov_inner_t).cpu().numpy().astype(bool)
    expand_np = ((combined > thresh_lo) & fov_inner_t).cpu().numpy().astype(bool)

    combined_np = combined.cpu().numpy()

    # Propagar semillas dentro de la región de expansión (reconstrucción morfológica)
    labeled_expand, _ = ndimage.label(expand_np)
    seed_labels = np.unique(labeled_expand[seeds_np])
    seed_labels = seed_labels[seed_labels > 0]
    vessel_np = np.isin(labeled_expand, seed_labels).astype(np.uint8)

    # Limpieza: quitamos motas pequeñas y cerramos solo huecos mínimos (disk 1)
    # para no engordar el trazado.
    vessel_np = morphology.remove_small_objects(vessel_np.astype(bool), min_size=min_object_size)
    vessel_np = morphology.binary_closing(vessel_np.astype(bool), disk(1))
    vessel_np = vessel_np.astype(np.uint8) * fov_inner

    del frangi_resp, tophat_resp, combined
    if USE_GPU:
        torch.cuda.empty_cache()

    return vessel_np, combined_np


# Alias para mantener compatibilidad con el resto del pipeline
segment_vessels = segment_vessels_gpu

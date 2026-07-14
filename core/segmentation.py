"""Segmentación de la red vascular.

Filtro de Frangi multiescala implementado en PyTorch (funciona en GPU o CPU)
combinado con top-hat morfológico. Basado en el notebook del TFM, con un
post-procesado más estricto (recorte circular del FOV y filtro de forma) para
ceñirse al trazado real del vaso y no contar bordes ni moteado de fondo.
"""

import numpy as np
import torch
import torch.nn.functional as F

from scipy import ndimage
from skimage import morphology
from skimage.morphology import disk, skeletonize
from skimage.measure import label as sk_label, regionprops
from skimage.filters import gaussian, threshold_otsu

from .device import DEVICE
from .preprocessing import tophat_gpu


def inner_fov_mask(fov_mask: np.ndarray, shrink: float = 0.95,
                   erosion: int = 6) -> np.ndarray:
    """Región de análisis interior al FOV.

    El borde del círculo del ojo (y los bloques brillantes que aparecen donde el
    círculo toca el marco de la imagen) se colaban como vaso y falseaban las
    métricas. Definimos la zona válida como un círculo concéntrico algo más
    pequeño que el FOV (recorte proporcional al radio, robusto al tamaño de la
    imagen), intersecado con el propio FOV y erosionado un poco.
    """
    ys, xs = np.where(fov_mask > 0)
    if len(xs) == 0:
        return fov_mask.astype(bool)
    cy, cx = ys.mean(), xs.mean()
    r_eq = np.sqrt(fov_mask.sum() / np.pi)
    H, W = fov_mask.shape
    yy, xx = np.ogrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r_eq * shrink) ** 2
    inner = circle & (fov_mask > 0)
    if erosion > 0:
        inner = morphology.binary_erosion(inner, disk(erosion))
    return inner


def keep_vessel_like(mask: np.ndarray, min_size: int = 60,
                     ecc_min: float = 0.85, large_area_factor: int = 6) -> np.ndarray:
    """Descarta manchas compactas (moteado, brillos) y conserva lo tubular.

    Para cada componente conexa se mira su tamaño y su forma:
      - Las componentes grandes (el árbol vascular principal) se conservan.
      - Las pequeñas solo se conservan si son alargadas (excentricidad alta),
        que es la firma de un vaso; las redondeadas (moteado) se eliminan.
    """
    labeled = sk_label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    large_area = large_area_factor * min_size
    for p in regionprops(labeled):
        if p.area < min_size:
            continue
        if p.area >= large_area or p.eccentricity >= ecc_min:
            out[labeled == p.label] = True
    return out


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
                        combined_weight: float = 0.7,
                        bg_sigma: float = 25.0,
                        seed_pct: float = 0.92,
                        body_mult: float = 1.0,
                        min_object_size: int = 60,
                        blob_ecc_min: float = 0.85,
                        fov_shrink: float = 0.95,
                        fov_erosion: int = 6,
                        min_hole_area: int = 1000) -> tuple:
    """Segmentación vascular que sigue la silueta real del vaso.

    Estrategia (dos señales con papeles distintos):
      - SEMILLAS (línea central del vaso): realce combinado Frangi + top-hat, que
        detecta estructuras tubulares tanto oscuras (Frangi) como claras (top-hat).
      - ANCHO del vaso: se define por el CONTRASTE de intensidad real respecto al
        fondo local (donde el vaso deja de destacar), no por la respuesta difusa
        del Frangi (que se extiende más allá del vaso y lo engordaba). El umbral es
        Otsu sobre ese contraste, así que la máscara se ciñe a la silueta real y
        una vena ancha se toma como un único vaso de su ancho, sin partirla por el
        reflejo central ni hincharla.
    """
    if device is None:
        device = DEVICE
    img_t = torch.tensor(img_clahe.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)
    fov_t = torch.tensor(fov_mask.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0).squeeze()

    frangi_resp = frangi_multiscale_gpu(img_clahe, fov_mask, device=device) * fov_t

    tophat_resp = tophat_gpu(img_t, kernel_size=25).squeeze() * fov_t
    th_max = tophat_resp.max()
    if th_max > 0:
        tophat_resp = tophat_resp / th_max

    # Realce combinado: Frangi capta vasos oscuros, top-hat capta los claros.
    combined = (combined_weight * frangi_resp + (1.0 - combined_weight) * tophat_resp) * fov_t
    combined_np = combined.cpu().numpy()

    # Región de análisis: interior circular del FOV (excluye el anillo del borde y
    # los bloques brillantes de los extremos).
    fov_inner = inner_fov_mask(fov_mask, shrink=fov_shrink, erosion=fov_erosion)
    fov_inner_t = torch.tensor(fov_inner.astype(np.float32), device=device).bool()

    # Semillas: puntos de línea central (realce combinado alto).
    seed_th = torch.quantile(combined[fov_inner_t], seed_pct)
    seeds_np = ((combined > seed_th) & fov_inner_t).cpu().numpy()

    # Ancho del vaso por contraste de intensidad frente al fondo local. Sirve para
    # vasos claros y oscuros (valor absoluto). Umbral adaptativo por Otsu.
    bg = gaussian(img_clahe, sigma=bg_sigma)
    dev = np.abs(img_clahe - bg)
    dv = dev[fov_inner]
    if dv.size == 0 or dv.max() <= 0:
        return np.zeros_like(fov_mask, dtype=np.uint8), combined_np
    try:
        otsu = threshold_otsu(dv)
    except Exception:
        otsu = float(dv.mean())
    body = (dev > otsu * body_mult) & fov_inner

    # Reconstrucción: conservamos solo las componentes del cuerpo del vaso que
    # contienen una semilla (así se descartan brillos/manchas sin línea central).
    labeled_body, _ = ndimage.label(body)
    seed_labels = np.unique(labeled_body[seeds_np])
    seed_labels = seed_labels[seed_labels > 0]
    vessel_np = np.isin(labeled_body, seed_labels)

    # Limpieza mínima: quitamos motas, tapamos pinchazos pequeños (no huecos
    # grandes, que son fondo real entre vasos) y descartamos manchas no tubulares.
    vessel_np = morphology.remove_small_objects(vessel_np, min_size=min_object_size)
    vessel_np = morphology.remove_small_holes(vessel_np, area_threshold=min_hole_area)
    vessel_np = keep_vessel_like(vessel_np, min_size=min_object_size, ecc_min=blob_ecc_min)
    vessel_np = (vessel_np & fov_inner).astype(np.uint8)

    del frangi_resp, tophat_resp, combined

    return vessel_np, combined_np


# Alias para mantener compatibilidad con el resto del pipeline
segment_vessels = segment_vessels_gpu

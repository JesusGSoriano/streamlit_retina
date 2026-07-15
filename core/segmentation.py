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
from skimage.filters import gaussian

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


def remove_thick_blobs(mask: np.ndarray, max_radius: float = 16.0) -> np.ndarray:
    """Borramos los núcleos gruesos (mácula, disco óptico, lesiones, brillos).

    Un vaso siempre es delgado: cada punto está cerca de un borde. Una mancha, en
    cambio, tiene un núcleo grueso, con un radio local (la transformada de
    distancia) mayor que el de cualquier vaso. Localizamos ese núcleo y lo
    borramos con su entorno píxel a píxel, no por componentes, para no llevarnos
    por delante los vasos finos que puedan tocar la mancha (por ejemplo en un
    cruce).
    """
    dist = ndimage.distance_transform_edt(mask)
    core = dist > max_radius
    if not core.any():
        return mask
    blob = morphology.binary_dilation(core, disk(int(round(max_radius)) + 2))
    return mask & ~blob


def keep_vessel_like(mask: np.ndarray, min_size: int = 55,
                     ecc_min: float = 0.80, solidity_max: float = 0.55) -> np.ndarray:
    """Conservamos lo tubular o ramificado y descartamos las manchas compactas.

    Para cada componente conexa: si es alargada (excentricidad alta) o ramificada
    (solidez baja) la damos por vaso y la conservamos; si es compacta y redondeada
    (moteado o restos de mancha) la descartamos.
    """
    labeled = sk_label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for p in regionprops(labeled):
        if p.area < min_size:
            continue
        if p.eccentricity >= ecc_min or p.solidity <= solidity_max:
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

    # Invertimos la imagen: así los vasos brillantes pasan a oscuros, que es lo
    # que espera Frangi.
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
                        seed_pct: float = 0.85,
                        expand_pct: float = 0.55,
                        bg_sigma: float = 25.0,
                        local_sigma: float = 51.0,
                        contrast_k: float = 1.3,
                        contrast_floor: float = 0.02,
                        fill_radius: int = 15,
                        min_object_size: int = 40,
                        blob_ecc_min: float = 0.80,
                        fov_shrink: float = 0.95,
                        fov_erosion: int = 6) -> tuple:
    """Segmentamos la red vascular combinando lo mejor de dos enfoques.

    Partimos de la DETECCIÓN del notebook: sobre el realce combinado (Frangi +
    top-hat) hacemos una histéresis permisiva (semillas p85, expansión p55) y
    propagamos por reconstrucción. Esto capta muchos vasos, incluidos los
    capilares finos, y coge bien la convergencia del disco óptico; además, como el
    Frangi es específico de estructuras tubulares, no marca la mácula ni las
    manchas suaves.

    Ese enfoque, por sí solo, parte las venas anchas en dos (el reflejo luminoso
    central deja un hueco) y no da un ancho ajustado. Para arreglarlo RELLENAMOS
    con el contraste de intensidad (dev = |imagen - fondo suave|), pero solo cerca
    de donde ya hay vaso detectado por Frangi. Así las venas anchas salen como un
    único vaso sólido de su ancho real, sin multiplicarlas, y sin reintroducir la
    mácula (que no está pegada a ningún vaso).

    Por último recortamos al interior circular del FOV (fuera bordes) y nos
    quedamos con lo tubular o ramificado. La respuesta combinada se devuelve para
    el módulo de fugas.
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

    # Realce combinado (Frangi para vasos oscuros, top-hat para los claros). Lo
    # devolvemos también para el módulo de fugas.
    combined = (combined_weight * frangi_resp + (1.0 - combined_weight) * tophat_resp) * fov_t
    combined_np = combined.cpu().numpy()

    # Región de análisis: interior circular del FOV (deja fuera el anillo del borde
    # y los bloques brillantes de los extremos).
    fov_inner = inner_fov_mask(fov_mask, shrink=fov_shrink, erosion=fov_erosion)

    # Detección del notebook: histéresis permisiva sobre el realce combinado.
    vals = combined_np[fov_inner]
    if vals.size == 0:
        return np.zeros_like(fov_mask, dtype=np.uint8), combined_np
    thresh_hi = np.quantile(vals, seed_pct)
    thresh_lo = np.quantile(vals, expand_pct)
    seeds = (combined_np > thresh_hi) & fov_inner
    expand = (combined_np > thresh_lo) & fov_inner
    labeled_expand, _ = ndimage.label(expand)
    seed_labels = np.unique(labeled_expand[seeds])
    seed_labels = seed_labels[seed_labels > 0]
    frangi_mask = np.isin(labeled_expand, seed_labels)

    # Relleno por contraste de intensidad, confirmado por cercanía a un vaso ya
    # detectado. Rellena y da el ancho real de las venas anchas (une el reflejo
    # central) sin colar la mácula.
    bg = gaussian(img_clahe, sigma=bg_sigma)
    dev = np.abs(img_clahe - bg)
    local = gaussian(dev, sigma=local_sigma)
    dev_mask = (dev > np.maximum(contrast_k * local, contrast_floor)) & fov_inner
    near_vessel = morphology.binary_dilation(frangi_mask, disk(fill_radius))
    vessel_np = frangi_mask | (dev_mask & near_vessel)

    # Limpieza: quitamos motas y nos quedamos con lo tubular/ramificado.
    vessel_np = morphology.remove_small_objects(vessel_np, min_size=min_object_size)
    vessel_np = morphology.binary_closing(vessel_np, disk(1))
    vessel_np = keep_vessel_like(vessel_np, min_size=min_object_size, ecc_min=blob_ecc_min)
    vessel_np = (vessel_np & fov_inner).astype(np.uint8)

    del frangi_resp, tophat_resp, combined

    return vessel_np, combined_np


# Alias para mantener compatibilidad con el resto del pipeline
segment_vessels = segment_vessels_gpu

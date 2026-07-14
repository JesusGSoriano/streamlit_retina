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
    """Elimina los núcleos gruesos (mácula, disco óptico, lesiones, brillos).

    Un vaso es siempre DELGADO: cada punto está cerca de un borde. Una mancha
    tiene un núcleo grueso, con radio local (transformada de distancia) mayor que
    cualquier vaso. Localizamos ese núcleo y lo borramos con su entorno, a nivel
    de PÍXEL (no de componente), para no arrastrar los vasos finos que estén
    conectados a la mancha (p.ej. por un cruce).
    """
    dist = ndimage.distance_transform_edt(mask)
    core = dist > max_radius
    if not core.any():
        return mask
    blob = morphology.binary_dilation(core, disk(int(round(max_radius)) + 2))
    return mask & ~blob


def keep_vessel_like(mask: np.ndarray, min_size: int = 55,
                     ecc_min: float = 0.80, solidity_max: float = 0.55) -> np.ndarray:
    """Conserva lo tubular/ramificado y descarta las manchas compactas.

    Para cada componente conexa:
      - Alargada (excentricidad alta) o ramificada (solidez baja): es un vaso o
        el árbol vascular -> se conserva.
      - Compacta y redondeada (moteado, restos de mancha) -> se descarta.
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
                        local_sigma: float = 51.0,
                        contrast_k: float = 1.3,
                        contrast_floor: float = 0.02,
                        min_object_size: int = 55,
                        blob_ecc_min: float = 0.80,
                        max_vessel_radius: float = 16.0,
                        fov_shrink: float = 0.95,
                        fov_erosion: int = 6,
                        min_hole_area: int = 1000) -> tuple:
    """Segmentación vascular que sigue la silueta real del vaso.

    El ancho de cada vaso lo define su CONTRASTE de intensidad frente al fondo
    local (dev = |imagen - fondo suave|), no la respuesta difusa del Frangi (que
    se extiende más allá del vaso y lo engordaba). Así la máscara se ciñe a la
    silueta y una vena ancha se toma como un único vaso sólido de su ancho (el
    reflejo luminoso central queda dentro), sin partirla ni hincharla. El valor
    absoluto hace que valga para vasos claros y oscuros.

    El umbral sobre el contraste es LOCAL y adaptativo: un píxel es vaso si su
    contraste supera contrast_k veces el contraste medio de su entorno. Donde el
    entorno es plano (zonas de capilares) el umbral baja y capta los vasos finos
    y tenues; junto a venas fuertes sube y no las engorda. Un umbral global no
    consigue ambas cosas a la vez, y un umbral demasiado alto en imágenes reales
    difusas no detectaba nada.

    El Frangi + top-hat se siguen calculando para el módulo de fugas (que recibe
    esta respuesta combinada).
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

    # Realce combinado (Frangi vasos oscuros, top-hat vasos claros). Se devuelve
    # para el módulo de fugas.
    combined = (combined_weight * frangi_resp + (1.0 - combined_weight) * tophat_resp) * fov_t
    combined_np = combined.cpu().numpy()

    # Región de análisis: interior circular del FOV (excluye el anillo del borde y
    # los bloques brillantes de los extremos).
    fov_inner = inner_fov_mask(fov_mask, shrink=fov_shrink, erosion=fov_erosion)

    # Contraste de intensidad frente al fondo local (vale para vasos claros y
    # oscuros: valor absoluto).
    bg = gaussian(img_clahe, sigma=bg_sigma)
    dev = np.abs(img_clahe - bg)

    # Umbral LOCAL adaptativo: contraste por encima de contrast_k veces el
    # contraste medio del entorno, con un suelo absoluto para no captar ruido en
    # zonas planas.
    local = gaussian(dev, sigma=local_sigma)
    thresh = np.maximum(contrast_k * local, contrast_floor)
    vessel_np = (dev > thresh) & fov_inner

    # Limpieza: quitamos motas, tapamos pinchazos pequeños (no huecos grandes, que
    # son fondo real entre vasos), borramos los núcleos gruesos no vasculares
    # (mácula, disco, lesiones) y descartamos manchas compactas no tubulares.
    vessel_np = morphology.remove_small_objects(vessel_np, min_size=min_object_size)
    vessel_np = morphology.remove_small_holes(vessel_np, area_threshold=min_hole_area)
    vessel_np = remove_thick_blobs(vessel_np, max_radius=max_vessel_radius)
    vessel_np = morphology.remove_small_objects(vessel_np, min_size=min_object_size)
    vessel_np = keep_vessel_like(vessel_np, min_size=min_object_size, ecc_min=blob_ecc_min)
    vessel_np = (vessel_np & fov_inner).astype(np.uint8)

    del frangi_resp, tophat_resp, combined

    return vessel_np, combined_np


# Alias para mantener compatibilidad con el resto del pipeline
segment_vessels = segment_vessels_gpu

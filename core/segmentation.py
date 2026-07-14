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
                        body_kernel: int = 41,
                        local_sigma: float = 41.0,
                        contrast_k: float = 2.0,
                        contrast_floor: float = 0.03,
                        min_object_size: int = 55,
                        blob_ecc_min: float = 0.80,
                        fov_shrink: float = 0.95,
                        fov_erosion: int = 6,
                        min_hole_area: int = 1000) -> tuple:
    """Segmentación vascular que sigue la silueta real del vaso.

    El ancho de cada vaso lo define su PROMINENCIA de intensidad frente al fondo
    (top-hat blanco + negro con un kernel mayor que el vaso más ancho), no la
    respuesta difusa del Frangi (que se extiende más allá del vaso y lo
    engordaba). El top-hat blanco capta vasos claros y el negro los oscuros, así
    que el ancho sale correcto para ambas polaridades; una vena ancha se toma
    como un único vaso sólido de su ancho (el reflejo luminoso central queda
    dentro), sin partirla ni hincharla.

    El umbral sobre la prominencia es LOCAL y adaptativo: un píxel es vaso si su
    prominencia supera varias veces la media de su entorno. Esto capta también
    los vasos finos y tenues (donde el entorno es plano) sin engordar las venas
    fuertes ni disparar el fondo, cosa que un umbral global no consigue.

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

    # Prominencia del vaso = top-hat blanco (vasos claros) + top-hat negro (vasos
    # oscuros), con un kernel mayor que el vaso más ancho para que el fondo no se
    # contamine con el propio vaso y el ancho salga correcto en ambas polaridades.
    pad = body_kernel // 2
    eroded = -F.max_pool2d(-img_t, body_kernel, stride=1, padding=pad)
    opening = F.max_pool2d(eroded, body_kernel, stride=1, padding=pad)
    white_hat = torch.clamp(img_t - opening, 0, 1)
    dilated = F.max_pool2d(img_t, body_kernel, stride=1, padding=pad)
    closing = -F.max_pool2d(-dilated, body_kernel, stride=1, padding=pad)
    black_hat = torch.clamp(closing - img_t, 0, 1)
    prominence = (white_hat + black_hat).squeeze().cpu().numpy()

    # Umbral LOCAL adaptativo: prominencia por encima de contrast_k veces la media
    # de su entorno, con un suelo absoluto para no captar ruido en zonas planas.
    local = gaussian(prominence, sigma=local_sigma)
    thresh = np.maximum(contrast_k * local, contrast_floor)
    vessel_np = (prominence > thresh) & fov_inner

    # Limpieza: quitamos motas, tapamos pinchazos pequeños (no huecos grandes, que
    # son fondo real entre vasos) y descartamos manchas compactas no tubulares.
    vessel_np = morphology.remove_small_objects(vessel_np, min_size=min_object_size)
    vessel_np = morphology.remove_small_holes(vessel_np, area_threshold=min_hole_area)
    vessel_np = keep_vessel_like(vessel_np, min_size=min_object_size, ecc_min=blob_ecc_min)
    vessel_np = (vessel_np & fov_inner).astype(np.uint8)

    del frangi_resp, tophat_resp, combined

    return vessel_np, combined_np


# Alias para mantener compatibilidad con el resto del pipeline
segment_vessels = segment_vessels_gpu

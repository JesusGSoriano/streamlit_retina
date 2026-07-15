"""Detección de fugas de fluoresceína.

Zonas hiperfluorescentes difusas (lagos irregulares) típicas de retinas enfermas,
excluyendo el disco óptico y la zona vascular central densa.

Portado literalmente del notebook del TFM.

Nota de fidelidad: en el pipeline, el argumento `frangi_response` recibe la
respuesta COMBINADA (0.7·frangi + 0.3·tophat) que devuelve `segment_vessels`,
no el Frangi puro. Se mantiene ese comportamiento idéntico al notebook.
"""

import numpy as np

from skimage import morphology
from skimage.morphology import disk
from skimage.measure import label as sk_label, regionprops
from skimage.filters import gaussian

from .segmentation import frangi_multiscale_gpu


def detect_optic_disc_mask(img_clahe: np.ndarray, fov_mask: np.ndarray,
                           frangi_response: np.ndarray = None) -> tuple:
    """Localiza el disco óptico por densidad vascular: es el punto donde convergen
    los vasos principales, así que es la zona de mayor densidad de la red. Nos
    apoyamos en la densidad y no solo en el brillo, que puede tener máximos
    sueltos fuera del disco. Devuelve el centro (cy, cx).
    """
    if frangi_response is not None:
        vessel_density = gaussian(frangi_response * fov_mask, sigma=35)
        brightness = gaussian(img_clahe * fov_mask, sigma=20)
        vd = vessel_density / (vessel_density.max() + 1e-8)
        br = brightness / (brightness.max() + 1e-8)
        score = 0.7 * vd + 0.3 * br
    else:
        score = gaussian(img_clahe * fov_mask, sigma=20)

    score = score * fov_mask
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return (cy, cx)


def compute_central_exclusion(img_clahe: np.ndarray, fov_mask: np.ndarray,
                              frangi_response: np.ndarray,
                              disc_center: tuple) -> np.ndarray:
    """Construye la máscara de exclusión del centro del ojo. El centro (disco óptico
    y las venas gruesas que salen de él) es una estructura constante y muy densa
    que genera falsos positivos de fuga, sobre todo en la zona justo alrededor y
    debajo del disco, donde las venas principales aún van muy juntas.

    Combinamos tres criterios y unimos lo que toca el disco para cubrir esa zona
    peridiscal de forma compacta:
      1. Círculo amplio alrededor del disco óptico.
      2. Zona de alta densidad vascular (venas gruesas convergentes).
      3. Zona de alto brillo sostenido (el propio disco y su halo).
    """
    H, W = img_clahe.shape
    cy, cx = disc_center

    # 1. Círculo alrededor del disco
    od_radius = int(0.26 * min(H, W))
    yy, xx = np.ogrid[:H, :W]
    disc_circle = ((xx - cx)**2 + (yy - cy)**2) < od_radius**2

    # 2. Alta densidad vascular
    vessel_density = gaussian(frangi_response * fov_mask, sigma=25)
    vd_vals = vessel_density[fov_mask > 0]
    dense = (vessel_density > np.percentile(vd_vals, 75)) & (fov_mask > 0)

    # 3. Alto brillo sostenido (halo peridiscal)
    brightness = gaussian(img_clahe * fov_mask, sigma=25)
    br_vals = brightness[fov_mask > 0]
    bright = (brightness > np.percentile(br_vals, 80)) & (fov_mask > 0)

    # Unimos densidad y brillo, y cerramos para compactar
    central = (dense | bright)
    central = morphology.binary_closing(central, disk(12))
    central = morphology.binary_dilation(central, disk(10))

    # Nos quedamos solo con la componente conexa que toca el disco óptico,
    # más el propio círculo. Así excluimos la masa central contigua al disco
    # (incluida la zona de debajo donde salen las venas) pero no manchas densas
    # legítimamente lejanas de la periferia.
    central_or_disc = central | disc_circle
    labeled = sk_label(central_or_disc)
    disc_label = labeled[cy, cx]
    if disc_label > 0:
        central_component = labeled == disc_label
    else:
        central_component = disc_circle

    exclusion = central_component | disc_circle
    exclusion = morphology.binary_closing(exclusion, disk(8))
    return exclusion


def detect_fluorescein_leakage(img_clahe: np.ndarray,
                               fov_mask: np.ndarray,
                               vessel_mask: np.ndarray = None,
                               frangi_response: np.ndarray = None,
                               min_leak_area: int = 150,
                               z_threshold: float = 3.0,
                               max_vesselness: float = 0.20,
                               min_solidity: float = 0.4,
                               max_eccentricity: float = 0.92) -> dict:
    """Detección de fugas de fluoresceína.

    Una fuga es una masa difusa e irregular donde la vasculatura ha perdido su
    forma tubular (un "lago" de fluorescencia), típica de los ratones enfermos.
    En un ojo sano no debería haber ninguna.

    El reto principal es no confundir fugas con dos cosas: los vasos normales, y
    sobre todo la zona central del ojo (disco óptico y venas principales que
    convergen). Para evitarlo:

    1. Excluimos toda la zona central de forma robusta: el disco óptico, su halo
       brillante y la masa de venas densas contigua, incluida la zona de debajo
       del disco donde las venas aún van muy juntas.
    2. Excluimos los vasos segmentados y un margen del borde del FOV.
    3. Trabajamos sobre el exceso de brillo local con umbral en z-score, así que
       si no hay nada anómalo no se detecta nada.
    4. Cada candidato debe tener vesselness baja (es difuso, no un vaso) y
       forma de mancha irregular, no filamentosa.
    """
    img = img_clahe.astype(np.float32) * fov_mask

    if frangi_response is None:
        frangi_response = frangi_multiscale_gpu(img_clahe, fov_mask).cpu().numpy()
    frangi_response = frangi_response * fov_mask

    H, W = img.shape

    # 1. Exclusión robusta del centro del ojo
    disc_center = detect_optic_disc_mask(img_clahe, fov_mask, frangi_response)
    central_exclusion = compute_central_exclusion(img_clahe, fov_mask, frangi_response, disc_center)

    # 2. Vasos y borde
    if vessel_mask is not None:
        vessel_exclusion = morphology.binary_dilation(vessel_mask.astype(bool), disk(5))
    else:
        vessel_exclusion = np.zeros_like(fov_mask, dtype=bool)

    fov_eroded = morphology.binary_erosion(fov_mask.astype(bool), disk(30))
    analysis_region = fov_eroded & ~vessel_exclusion & ~central_exclusion

    empty = {
        'n_leaks': 0, 'total_leak_area_px': 0, 'mean_leak_area_px': 0.0,
        'leakage_mask': np.zeros_like(fov_mask, dtype=np.uint8), 'leak_props': [],
        'brightness_threshold': 0.0,
    }
    if analysis_region.sum() < 500:
        return empty

    # 3. Exceso de brillo local + umbral z-score
    local_mean = gaussian(img, sigma=40)
    local_excess = img - local_mean

    vals = local_excess[analysis_region]
    mu = float(np.mean(vals))
    sig = float(np.std(vals))
    if sig < 1e-6:
        return empty

    thresh = mu + z_threshold * sig

    candidate = (local_excess > thresh) & analysis_region
    candidate = morphology.remove_small_objects(candidate, min_size=min_leak_area)
    candidate = morphology.binary_closing(candidate, disk(4))
    candidate = morphology.remove_small_objects(candidate, min_size=min_leak_area)

    if candidate.sum() == 0:
        return {**empty, 'brightness_threshold': float(thresh)}

    frangi_smooth = gaussian(frangi_response, sigma=3)

    labeled = sk_label(candidate)
    props = regionprops(labeled, intensity_image=img)

    final_mask = np.zeros_like(labeled, dtype=np.uint8)
    valid_leaks = []

    for p in props:
        if p.area < min_leak_area:
            continue
        region_mask = labeled == p.label

        # 4a. La región no tiene estructura tubular (vesselness baja)
        mean_vesselness = frangi_smooth[region_mask].mean()
        is_diffuse = mean_vesselness < max_vesselness

        # 4b. Forma de mancha compacta no filamentosa
        is_blob = (p.solidity > min_solidity) and (p.eccentricity < max_eccentricity)

        # 4c. Contraste claro respecto al entorno
        is_strong = (p.mean_intensity - local_mean[region_mask].mean()) > 1.2 * sig

        if is_diffuse and is_blob and is_strong:
            final_mask[labeled == p.label] = 1
            valid_leaks.append(p)

    return {
        'n_leaks': len(valid_leaks),
        'total_leak_area_px': int(final_mask.sum()),
        'mean_leak_area_px': float(np.mean([p.area for p in valid_leaks])) if valid_leaks else 0.0,
        'leakage_mask': final_mask,
        'leak_props': valid_leaks,
        'brightness_threshold': float(thresh),
    }

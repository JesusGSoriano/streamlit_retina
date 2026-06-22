"""Métricas vasculares.

  - Área / densidad vascular (% del FOV ocupado por vasos).
  - Calibre (mediana de 2x la transformada de distancia sobre los puntos internos
    del esqueleto, excluyendo nodos de cruce/bifurcación).
  - Número de segmentos vasculares.
  - Tortuosidad media sobre trayectos largos (longitud real / distancia euclídea,
    ponderada por longitud del segmento).
  - Longitud total de la red.

Portado del notebook del TFM.
"""

import numpy as np

from scipy import ndimage
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


def skeleton_neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    """Cuenta los vecinos (conectividad 8) de cada píxel del esqueleto.

    Devuelve un array con el número de vecinos por píxel de esqueleto (0 fuera
    del esqueleto). Sirve para distinguir puntos internos del vaso (2 vecinos)
    de los nodos de cruce/bifurcación (3+ vecinos).
    """
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = ndimage.convolve(skeleton.astype(np.uint8), k, mode='constant')
    return neigh * skeleton



def compute_vascular_area(vessel_mask: np.ndarray, fov_mask: np.ndarray) -> dict:
    vessel_area = int(vessel_mask.sum())
    fov_area = int(fov_mask.sum())
    density = 100.0 * vessel_area / fov_area if fov_area > 0 else 0.0

    return {
        'vessel_area_px': vessel_area,
        'fov_area_px': fov_area,
        'vessel_density_pct': density,
    }


def compute_skeleton_metrics(vessel_mask: np.ndarray) -> dict:
    if vessel_mask.sum() == 0:
        return {
            'n_vessel_segments': 0,
            'calibre_px': np.nan,
            'std_diameter_px': np.nan,
            'mean_tortuosity': np.nan,
            'std_tortuosity': np.nan,
            'total_length_px': 0,
            'vessel_diameters': [],
            'vessel_tortuosities': [],
        }

    binary = vessel_mask.astype(bool)
    dist_transform = ndimage.distance_transform_edt(binary)
    skeleton = skeletonize(binary)

    try:
        skel_obj = Skeleton(skeleton)
        branch_data = summarize(skel_obj, separator='-')

        valid_branches = branch_data[branch_data['branch-type'] > 0].copy()
        # Filtramos ramas muy cortas, que dan tortuosidades ruidosas
        valid_branches = valid_branches[valid_branches['branch-distance'] >= 5]

        # Tortuosidad solo sobre trayectos largos (>= 30 px): los segmentos
        # cortos son casi rectos y dan valores poco informativos (~1.08). Los
        # trayectos largos capturan mejor el serpenteo real del vaso.
        long_branches = valid_branches[valid_branches['branch-distance'] >= 30]
        tortuosities = []
        seg_lengths = []
        for _, row in long_branches.iterrows():
            euc = row.get('euclidean-distance', 0.0)
            brn = row.get('branch-distance', 0.0)
            # La tortuosidad es longitud_real / distancia_en_linea_recta (>= 1).
            # Pedimos euc > 1 px para evitar dividir por distancias diminutas que
            # disparan el ratio de forma artificial.
            if euc > 1.0 and brn >= euc:
                tortuosities.append(brn / euc)
                seg_lengths.append(brn)

        # Calibre: mediana de 2x la transformada de distancia sobre los puntos
        # INTERNOS del esqueleto (exactamente 2 vecinos), excluyendo los nodos de
        # cruce/bifurcación (3+ vecinos) y el disco óptico, donde la transformada
        # de distancia se dispara e infla el calibre.
        neigh = skeleton_neighbor_count(skeleton)
        internal = skeleton & (neigh == 2)
        diam_internal = 2 * dist_transform[np.where(internal)]
        calibre_px = float(np.median(diam_internal)) if len(diam_internal) > 0 else np.nan
        std_diameter = float(np.std(diam_internal)) if len(diam_internal) > 0 else np.nan

        total_length = int(skeleton.sum())
        n_segments = len(valid_branches)

        # Media de tortuosidad ponderada por longitud del segmento: los vasos
        # largos (más fiables) pesan más que los fragmentos cortos.
        if tortuosities:
            tort_arr = np.array(tortuosities)
            len_arr = np.array(seg_lengths)
            mean_tort = float(np.average(tort_arr, weights=len_arr))
            std_tort = float(np.sqrt(np.average((tort_arr - mean_tort)**2, weights=len_arr)))
        else:
            mean_tort = np.nan
            std_tort = np.nan

        return {
            'n_vessel_segments': n_segments,
            'calibre_px': calibre_px,
            'std_diameter_px': std_diameter,
            'mean_tortuosity': mean_tort,
            'std_tortuosity': std_tort,
            'total_length_px': total_length,
            'vessel_diameters': diam_internal.tolist(),
            'vessel_tortuosities': tortuosities,
            'skeleton': skeleton,
            'dist_transform': dist_transform,
        }

    except Exception:
        neigh = skeleton_neighbor_count(skeleton)
        internal = skeleton & (neigh == 2)
        diam_internal = 2 * dist_transform[np.where(internal)]
        calibre_px = float(np.median(diam_internal)) if len(diam_internal) > 0 else np.nan
        std_diameter = float(np.std(diam_internal)) if len(diam_internal) > 0 else np.nan
        total_length = int(skeleton.sum())
        labeled_skel, n_segments = ndimage.label(skeleton)

        return {
            'n_vessel_segments': n_segments,
            'calibre_px': calibre_px,
            'std_diameter_px': std_diameter,
            'mean_tortuosity': np.nan,
            'std_tortuosity': np.nan,
            'total_length_px': total_length,
            'vessel_diameters': diam_internal.tolist(),
            'vessel_tortuosities': [],
            'skeleton': skeleton,
            'dist_transform': dist_transform,
        }

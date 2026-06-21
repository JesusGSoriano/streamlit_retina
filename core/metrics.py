"""Métricas vasculares.

  - Área / densidad vascular (% del FOV ocupado por vasos).
  - Calibre medio (2x transformada de distancia sobre el esqueleto).
  - Número de segmentos vasculares.
  - Tortuosidad media (longitud real / distancia euclídea, ponderada por longitud).
  - Longitud total de la red.

Portado literalmente del notebook del TFM.
"""

import numpy as np

from scipy import ndimage
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


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
            'mean_diameter_px': np.nan,
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

        tortuosities = []
        seg_lengths = []
        for _, row in valid_branches.iterrows():
            euc = row.get('euclidean-distance', 0.0)
            brn = row.get('branch-distance', 0.0)
            # La tortuosidad es longitud_real / distancia_en_linea_recta (>= 1).
            # Pedimos euc > 1 px para evitar dividir por distancias diminutas que
            # disparan el ratio de forma artificial.
            if euc > 1.0 and brn >= euc:
                tortuosities.append(brn / euc)
                seg_lengths.append(brn)

        skeleton_coords = np.where(skeleton)
        diameters = 2 * dist_transform[skeleton_coords]

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
            'mean_diameter_px': float(np.mean(diameters)) if len(diameters) > 0 else np.nan,
            'std_diameter_px': float(np.std(diameters)) if len(diameters) > 0 else np.nan,
            'mean_tortuosity': mean_tort,
            'std_tortuosity': std_tort,
            'total_length_px': total_length,
            'vessel_diameters': diameters.tolist(),
            'vessel_tortuosities': tortuosities,
            'skeleton': skeleton,
            'dist_transform': dist_transform,
        }

    except Exception:
        skeleton_coords = np.where(skeleton)
        diameters = 2 * dist_transform[skeleton_coords]
        total_length = int(skeleton.sum())
        labeled_skel, n_segments = ndimage.label(skeleton)

        return {
            'n_vessel_segments': n_segments,
            'mean_diameter_px': float(np.mean(diameters)) if len(diameters) > 0 else np.nan,
            'std_diameter_px': float(np.std(diameters)) if len(diameters) > 0 else np.nan,
            'mean_tortuosity': np.nan,
            'std_tortuosity': np.nan,
            'total_length_px': total_length,
            'vessel_diameters': diameters.tolist(),
            'vessel_tortuosities': [],
            'skeleton': skeleton,
            'dist_transform': dist_transform,
        }

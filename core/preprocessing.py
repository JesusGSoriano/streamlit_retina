"""Preprocesamiento de imágenes retinianas (angiografía de fluoresceína).

Lo trasladamos tal cual del notebook del TFM. Hacemos tres cosas:
  1. Extraemos el canal verde y lo normalizamos a [0, 1].
  2. Calculamos la máscara FOV (la región circular del ojo).
  3. Aplicamos CLAHE (ecualización adaptativa de contraste) y la limitamos al FOV.

Las funciones reciben rutas o arrays y devuelven arrays. No pintan nada.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.morphology import disk
from skimage.measure import label as sk_label, regionprops


def create_fov_mask(img_gray: np.ndarray, threshold_factor: float = 0.05) -> np.ndarray:
    thresh = img_gray.max() * threshold_factor
    mask = img_gray > thresh
    mask = binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, disk(15))
    labeled = sk_label(mask)
    if labeled.max() > 0:
        props = regionprops(labeled)
        largest = max(props, key=lambda p: p.area)
        mask = labeled == largest.label
    return mask.astype(np.uint8)


def clahe_gpu(img_uint8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def tophat_gpu(img_tensor: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
    """Top-hat morfológico usando PyTorch.

    img_tensor: (1,1,H,W) float32 en el dispositivo.
    """
    # Erosión = -max_pool(-x)
    pad = kernel_size // 2
    eroded = -F.max_pool2d(-img_tensor, kernel_size, stride=1, padding=pad)
    # Dilatación de la erosión = apertura
    opened = F.max_pool2d(eroded, kernel_size, stride=1, padding=pad)
    # Top-hat = original - apertura
    tophat = img_tensor - opened
    return torch.clamp(tophat, 0, 1)


def preprocess_retinal_image_gpu(path: str):
    """Preprocesamiento de una imagen retiniana.

    Devuelve todo en numpy (CPU) para compatibilidad con el resto del pipeline:
        img_rgb, img_green, img_clahe, fov_mask
    """
    img_rgb = np.array(Image.open(path).convert('RGB'))

    # Canal verde
    img_green = img_rgb[:, :, 1].astype(np.float32) / 255.0

    # Máscara FOV (CPU, rápido)
    fov_mask = create_fov_mask(img_green)

    # CLAHE (CPU con OpenCV, ya óptimo)
    img_uint8 = (img_green * 255).astype(np.uint8)
    img_clahe_uint8 = clahe_gpu(img_uint8)
    img_clahe = img_clahe_uint8.astype(np.float32) / 255.0
    img_clahe = img_clahe * fov_mask

    return img_rgb, img_green, img_clahe, fov_mask


# Alias para mantener compatibilidad con el resto del pipeline
preprocess_retinal_image = preprocess_retinal_image_gpu

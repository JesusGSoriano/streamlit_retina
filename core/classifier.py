"""Clasificador Control frente a db/db (ensemble de 5 ResNet18).

Cargamos el ensemble entrenado (`clasificador_retina_ensemble.pt`) y hacemos la
inferencia. Nos mantenemos fieles al notebook del TFM:

  - La entrada al modelo es el canal verde crudo de la imagen, replicado a tres
    canales (no la imagen RGB ni la CLAHE).
  - Transforms de test: redimensionamos a 224x224, pasamos a tensor y
    normalizamos con media 0.5 y desviación 0.5.
  - Sacamos el softmax de cada modelo, promediamos las probabilidades de los
    cinco y decidimos con el umbral guardado (0.5). La clase 1 es db/db.

Construimos la arquitectura con weights=None (sin descargar ImageNet) porque el
state_dict del ensemble ya define todos los pesos; el resultado de la inferencia
es idéntico al del notebook.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from .device import DEVICE


def build_model() -> nn.Module:
    """Arquitectura idéntica a la del notebook: ResNet18 con la capa fc reemplazada."""
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(64, 2),
    )
    return model


def build_test_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


@dataclass
class RetinaEnsemble:
    """Ensemble cargado y listo para inferencia."""
    models: list          # lista de nn.Module en eval()
    transform: transforms.Compose
    threshold: float
    img_size: int
    seeds: list
    animals: list
    device: torch.device
    temperature: float = 1.0   # temperature scaling; 1.0 = sin calibrar


def load_ensemble(model_path: str, device: torch.device = None) -> RetinaEnsemble:
    """Cargamos el diccionario .pt y reconstruimos los cinco modelos del ensemble.

    La temperatura de calibración (temperature scaling) se toma, por orden, de la
    variable de entorno RETINA_TEMPERATURE, de la clave 'temperature' del .pt, o
    1.0 si no hay ninguna (sin calibrar).
    """
    if device is None:
        device = DEVICE

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state_dicts = ckpt['ensemble_state_dicts']
    img_size = int(ckpt.get('img_size', 224))
    threshold = float(ckpt.get('threshold', 0.5))
    seeds = list(ckpt.get('seeds', []))
    animals = list(ckpt.get('animals', []))
    temperature = float(os.environ.get('RETINA_TEMPERATURE', ckpt.get('temperature', 1.0)))

    loaded_models = []
    for sd in state_dicts:
        model = build_model()
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        loaded_models.append(model)

    return RetinaEnsemble(
        models=loaded_models,
        transform=build_test_transform(img_size),
        threshold=threshold,
        img_size=img_size,
        seeds=seeds,
        animals=animals,
        device=device,
        temperature=temperature,
    )


def _to_model_input(img_rgb: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    """Reproducimos el preprocesado del RetinaDataset del notebook: tomamos el
    canal verde, lo pasamos a una imagen en gris, lo replicamos a tres canales y
    le aplicamos los transforms.
    """
    img_green = Image.fromarray(np.asarray(img_rgb)[:, :, 1])
    img_rgb_gray = img_green.convert('RGB')
    return transform(img_rgb_gray)


@torch.no_grad()
def classify_image(ensemble: RetinaEnsemble, img_rgb: np.ndarray) -> dict:
    """Clasificamos una imagen (array RGB) con el ensemble.

    Estos modelos, de fábrica, dan probabilidades sobreconfiadas (pegadas a 0 o a
    1), así que sin calibrar no reflejan la confianza real. Lo corregimos con
    temperature scaling: dividimos el logit de la probabilidad del ensemble por
    una temperatura T (T > 1 acerca la probabilidad a 0.5). Con T = 1.0 no se
    calibra nada. T se ajusta sobre datos de validación (ver README).

    Devolvemos:
        prob_dbdb : probabilidad media de la clase db/db (clase 1) [0, 1].
        pred      : 0 = Control, 1 = db/db.
        label     : 'Control' o 'db/db (Enfermo)'.
        per_model_probs : probabilidad db/db de cada uno de los 5 modelos.
        n_agree   : cuántos de los modelos coinciden con el veredicto del ensemble.
        n_models  : número de modelos del ensemble.
        temperature : temperatura de calibración aplicada.
    """
    x = _to_model_input(img_rgb, ensemble.transform).unsqueeze(0).to(ensemble.device)
    T = max(float(ensemble.temperature), 1e-6)

    # Probabilidad cruda de cada modelo (sin calibrar). La usamos para el acuerdo
    # y el rango del ensemble.
    per_model_probs = []
    for model in ensemble.models:
        prob_db = torch.softmax(model(x), dim=1)[0, 1].item()
        per_model_probs.append(prob_db)

    prob_raw = float(np.mean(per_model_probs))

    # Calibración por temperatura sobre el logit de la probabilidad del ensemble:
    # con T > 1 la acercamos hacia 0.5, corrigiendo la sobreconfianza. T se ajusta
    # sobre datos de validación (ver README).
    p = min(max(prob_raw, 1e-6), 1 - 1e-6)
    logit = np.log(p / (1 - p))
    prob_dbdb = float(1.0 / (1.0 + np.exp(-logit / T)))

    pred = int(prob_dbdb >= ensemble.threshold)
    label = 'db/db (Enfermo)' if pred == 1 else 'Control'

    # Acuerdo del ensemble: cuántos modelos, por separado, deciden lo mismo que la
    # media. Es una señal de robustez que no depende de la calibración.
    n_agree = int(sum(1 for pm in per_model_probs
                      if int(pm >= ensemble.threshold) == pred))

    return {
        'prob_dbdb': prob_dbdb,
        'pred': pred,
        'label': label,
        'per_model_probs': per_model_probs,
        'n_agree': n_agree,
        'n_models': len(ensemble.models),
        'temperature': T,
    }

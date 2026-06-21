"""Clasificador Control vs db/db (ensemble de 5 ResNet18).

Carga del ensemble entrenado (`clasificador_retina_ensemble.pt`) e inferencia.

Fidelidad con el notebook del TFM:
  - La entrada al modelo es el CANAL VERDE crudo de la imagen, replicado a 3
    canales (no la imagen RGB ni la CLAHE).
  - Transforms de test: Resize(224,224) -> ToTensor -> Normalize(mean=[0.5], std=[0.5]).
  - Se hace softmax de cada modelo, se promedian las probabilidades de los 5
    modelos y se decide con el threshold guardado (0.5). La clase 1 es db/db.

Nota: la arquitectura se construye con weights=None (sin descargar ImageNet),
porque el state_dict del ensemble define todos los pesos. El resultado de
inferencia es idéntico al del notebook.
"""

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


def load_ensemble(model_path: str, device: torch.device = None) -> RetinaEnsemble:
    """Carga el diccionario .pt y reconstruye los 5 modelos del ensemble."""
    if device is None:
        device = DEVICE

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state_dicts = ckpt['ensemble_state_dicts']
    img_size = int(ckpt.get('img_size', 224))
    threshold = float(ckpt.get('threshold', 0.5))
    seeds = list(ckpt.get('seeds', []))
    animals = list(ckpt.get('animals', []))

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
    )


def _to_model_input(img_rgb: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    """Replica exactamente el preprocesado del RetinaDataset del notebook:
    canal verde -> PIL en gris -> replicado a 3 canales -> transforms.
    """
    img_green = Image.fromarray(np.asarray(img_rgb)[:, :, 1])
    img_rgb_gray = img_green.convert('RGB')
    return transform(img_rgb_gray)


@torch.no_grad()
def classify_image(ensemble: RetinaEnsemble, img_rgb: np.ndarray) -> dict:
    """Clasifica una imagen (array RGB) con el ensemble.

    Devuelve:
        prob_dbdb : probabilidad media de la clase db/db (clase 1) [0, 1].
        pred      : 0 = Control, 1 = db/db.
        label     : 'Control' o 'db/db (Enfermo)'.
        per_model_probs : probabilidad db/db de cada uno de los 5 modelos.
    """
    x = _to_model_input(img_rgb, ensemble.transform).unsqueeze(0).to(ensemble.device)

    per_model_probs = []
    for model in ensemble.models:
        prob_db = torch.softmax(model(x), dim=1)[0, 1].item()
        per_model_probs.append(prob_db)

    prob_dbdb = float(np.mean(per_model_probs))
    pred = int(prob_dbdb >= ensemble.threshold)
    label = 'db/db (Enfermo)' if pred == 1 else 'Control'

    return {
        'prob_dbdb': prob_dbdb,
        'pred': pred,
        'label': label,
        'per_model_probs': per_model_probs,
    }

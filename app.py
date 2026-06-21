"""Demo Streamlit — Análisis automático de imágenes retinianas (TFM / CIPF).

Pipeline de análisis de angiografías de fluoresceña de modelos murinos para el
estudio de la retinopatía diabética:
  1. Preprocesado (canal verde, máscara FOV, CLAHE).
  2. Segmentación vascular (Frangi multiescala + top-hat).
  3. Métricas vasculares (densidad, calibre, segmentos, tortuosidad, longitud).
  4. Detección de fugas de fluoresceína.
  5. Clasificación Control vs db/db (ensemble de 5 ResNet18).

La lógica vive en `core/`; este archivo solo se encarga de la interfaz.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from core.device import DEVICE, USE_GPU
from core.pipeline import analyze_image
from core.classifier import load_ensemble, classify_image

# ── Configuración ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    'RETINA_MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'models', 'clasificador_retina_ensemble.pt'),
)

VESSEL_COLOR = [0, 100, 255]    # azul (igual que el notebook)
LEAK_COLOR = [255, 220, 0]      # amarillo (igual que el notebook)

st.set_page_config(
    page_title='Análisis Retiniano — TFM/CIPF',
    page_icon='👁️',
    layout='wide',
)


# ── Carga del ensemble (una sola vez) ────────────────────────────────────────
@st.cache_resource(show_spinner='Cargando ensemble de clasificación…')
def get_ensemble(path: str):
    if not os.path.exists(path):
        return None
    return load_ensemble(path, device=DEVICE)


def make_vessel_overlay(img_rgb: np.ndarray, vessel_mask: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    overlay[vessel_mask > 0] = VESSEL_COLOR
    return overlay


def make_leak_overlay(img_rgb: np.ndarray, vessel_mask: np.ndarray,
                      leakage_mask: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    overlay[vessel_mask > 0] = VESSEL_COLOR
    overlay[leakage_mask > 0] = LEAK_COLOR
    return overlay


def metrics_dataframe(res: dict) -> pd.DataFrame:
    def fmt(v, suffix='', dec=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        if isinstance(v, float):
            return f'{v:.{dec}f}{suffix}'
        return f'{v:,}{suffix}'

    rows = [
        ('Densidad vascular', fmt(res.get('vessel_density_pct'), ' %')),
        ('Nº segmentos vasculares', fmt(res.get('n_vessel_segments'))),
        ('Calibre medio', fmt(res.get('mean_diameter_px'), ' px')),
        ('Tortuosidad media', fmt(res.get('mean_tortuosity'), '', 3)),
        ('Longitud total de la red', fmt(res.get('total_length_px'), ' px')),
        ('Nº fugas de fluoresceína', fmt(res.get('n_leaks'))),
        ('Área total de fugas', fmt(res.get('total_leak_area_px'), ' px')),
    ]
    return pd.DataFrame(rows, columns=['Métrica', 'Valor'])


def analyze_uploaded_file(uploaded_file, ensemble):
    """Guarda el fichero subido en disco y ejecuta pipeline + clasificación."""
    suffix = os.path.splitext(uploaded_file.name)[1] or '.tif'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        res = analyze_image(tmp_path)
        clf = None
        if res['error'] is None and ensemble is not None:
            clf = classify_image(ensemble, res['img_rgb'])
    finally:
        os.unlink(tmp_path)

    return res, clf


def render_result(name: str, res: dict, clf: dict, ensemble):
    st.subheader(f'📄 {name}')

    if res['error'] is not None:
        st.error(f'No se pudo procesar la imagen: {res["error"]}')
        return

    # ── Veredicto del clasificador ──
    if ensemble is None:
        st.warning(
            'Modelo de clasificación no disponible — se muestran solo las '
            'métricas del pipeline. (Coloca `clasificador_retina_ensemble.pt` '
            'en la carpeta `models/`.)'
        )
    elif clf is not None:
        prob = clf['prob_dbdb']
        if clf['pred'] == 1:
            st.error(f'### 🔴 Veredicto: **db/db (Enfermo)**')
        else:
            st.success(f'### 🟢 Veredicto: **Control (Sano)**')
        c1, c2 = st.columns(2)
        c1.metric('Probabilidad db/db', f'{prob*100:.1f} %')
        c2.metric('Probabilidad Control', f'{(1-prob)*100:.1f} %')
        st.progress(prob)
        st.caption(
            'Probabilidad media del ensemble de 5 modelos. '
            f'Umbral de decisión: {ensemble.threshold:.2f}. '
            f'Probabilidades por modelo: '
            + ', '.join(f'{p:.2f}' for p in clf['per_model_probs'])
        )

    # ── Imágenes ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(res['img_rgb'], caption='Original', use_container_width=True)
    with col2:
        st.image(
            make_vessel_overlay(res['img_rgb'], res['vessel_mask']),
            caption='Red vascular segmentada (azul)', use_container_width=True,
        )
    with col3:
        st.image(
            make_leak_overlay(res['img_rgb'], res['vessel_mask'], res['leakage_mask']),
            caption=f'Fugas detectadas (amarillo) — N={res.get("n_leaks", 0)}',
            use_container_width=True,
        )

    # ── Métricas ──
    st.markdown('**Métricas vasculares**')
    st.dataframe(metrics_dataframe(res), hide_index=True, use_container_width=True)

    st.divider()


def main():
    st.title('👁️ Análisis automático de imágenes retinianas')
    st.markdown(
        'Demo del pipeline del TFM (en colaboración con el **CIPF**) para el '
        'análisis de angiografías de fluoresceína de modelos murinos en el '
        'estudio de la **retinopatía diabética**.'
    )

    ensemble = get_ensemble(MODEL_PATH)

    # ── Barra lateral ──
    with st.sidebar:
        st.header('ℹ️ Información')
        st.write(f'**Dispositivo:** {"GPU (CUDA)" if USE_GPU else "CPU"}')
        if ensemble is not None:
            st.success('Modelo de clasificación cargado ✓')
            st.caption(
                f'Ensemble de {len(ensemble.models)} ResNet18 · '
                f'img_size={ensemble.img_size} · threshold={ensemble.threshold}'
            )
        else:
            st.error('Modelo de clasificación no encontrado')
            st.caption(f'Ruta esperada: `{MODEL_PATH}`')
        st.divider()
        st.markdown(
            '**Pasos del pipeline:**\n'
            '1. Preprocesado (canal verde + FOV + CLAHE)\n'
            '2. Segmentación vascular (Frangi + top-hat)\n'
            '3. Métricas vasculares\n'
            '4. Detección de fugas\n'
            '5. Clasificación Control vs db/db'
        )

    uploaded_files = st.file_uploader(
        'Sube una o varias imágenes de fondo de ojo (.tif)',
        type=['tif', 'tiff'],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info('⬆️ Sube al menos una imagen `.tif` para comenzar el análisis.')
        return

    for uploaded_file in uploaded_files:
        with st.spinner(f'Analizando {uploaded_file.name}…'):
            res, clf = analyze_uploaded_file(uploaded_file, ensemble)
        render_result(uploaded_file.name, res, clf, ensemble)


if __name__ == '__main__':
    main()

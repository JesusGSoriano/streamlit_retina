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
import urllib.request

import streamlit as st

from core.device import DEVICE, USE_GPU
from core.pipeline import analyze_image
from core.classifier import load_ensemble, classify_image

# Configuración
MODEL_FILENAME = 'clasificador_retina_ensemble.pt'

MODEL_PATH = os.environ.get(
    'RETINA_MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'models', MODEL_FILENAME),
)

# URL pública del modelo (asset del GitHub Release). Sirve de valor por defecto
# para que la app descargue el modelo automáticamente en el despliegue sin tener
# que configurar nada. Se puede sobreescribir con RETINA_MODEL_URL.
DEFAULT_MODEL_URL = (
    'https://github.com/JesusGSoriano/streamlit_retina/releases/download/'
    'v1.0/clasificador_retina_ensemble.pt'
)

# Ruta de descarga garantizada como escribible (en Streamlit Cloud la carpeta
# del repo puede no serlo).
DOWNLOAD_PATH = os.path.join(tempfile.gettempdir(), MODEL_FILENAME)


def get_model_url() -> str:
    """URL desde la que descargar el modelo si no está en disco.

    Orden de preferencia: variable de entorno RETINA_MODEL_URL > secreto de
    Streamlit (st.secrets) > URL por defecto del GitHub Release.
    """
    url = os.environ.get('RETINA_MODEL_URL', '')
    if not url:
        try:
            url = st.secrets.get('RETINA_MODEL_URL', '')
        except Exception:
            url = ''
    return url or DEFAULT_MODEL_URL

VESSEL_COLOR = [0, 100, 255]    # azul (igual que el notebook)
LEAK_COLOR = [255, 220, 0]      # amarillo (igual que el notebook)

st.set_page_config(
    page_title='Análisis de imágenes retinianas — TFM/CIPF',
    layout='wide',
)


# Carga del ensemble (una sola vez)
def _download_model(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with st.spinner('Descargando el modelo (~220 MB, solo la primera vez)…'):
        urllib.request.urlretrieve(url, dst)


@st.cache_resource(show_spinner='Cargando ensemble de clasificación…')
def get_ensemble():
    # 1. ¿Está ya en disco? (local o vía Git LFS, o descargado antes)
    for path in (MODEL_PATH, DOWNLOAD_PATH):
        if os.path.exists(path):
            return load_ensemble(path, device=DEVICE)

    # 2. Descargar desde la URL (release/HF) a una ruta escribible
    url = get_model_url()
    if not url:
        return None
    try:
        _download_model(url, DOWNLOAD_PATH)
    except Exception as e:
        st.error(f'No se pudo descargar el modelo desde {url}\n\n{e}')
        return None
    return load_ensemble(DOWNLOAD_PATH, device=DEVICE)


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
        ('Calibre (mediana, px)', fmt(res.get('calibre_px'), ' px')),
        ('Tortuosidad (trayecto largo)', fmt(res.get('mean_tortuosity'), '', 3)),
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
    st.subheader(name)

    if res['error'] is not None:
        st.error(f'No se pudo procesar la imagen: {res["error"]}')
        return

    # Clasificación
    if ensemble is None:
        st.warning(
            'Modelo de clasificación no disponible; se muestran solo las '
            'métricas del pipeline.'
        )
    elif clf is not None:
        prob = clf['prob_dbdb']
        label = 'db/db (enfermo)' if clf['pred'] == 1 else 'Control (sano)'
        if clf['pred'] == 1:
            st.error(f'Clasificación: **{label}**')
        else:
            st.success(f'Clasificación: **{label}**')
        c1, c2 = st.columns(2)
        c1.metric('Probabilidad db/db', f'{prob*100:.1f} %')
        c2.metric('Probabilidad Control', f'{(1-prob)*100:.1f} %')
        st.progress(prob)
        st.caption(
            'Probabilidad media del ensemble de 5 modelos. '
            f'Umbral de decisión: {ensemble.threshold:.2f}. '
            'Probabilidades por modelo: '
            + ', '.join(f'{p:.2f}' for p in clf['per_model_probs'])
        )

    # Imágenes
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

    # Métricas
    st.markdown('**Métricas vasculares**')
    mdf = metrics_dataframe(res)
    st.dataframe(mdf, hide_index=True, use_container_width=True)
    with st.expander('Copiar métricas al portapapeles'):
        st.caption(
            'Pulse el icono de copiar (esquina superior derecha del recuadro) y '
            'pegue en Excel o en una hoja de cálculo: cada métrica y su valor '
            'quedan en columnas separadas.'
        )
        st.code(mdf.to_csv(sep='\t', index=False), language=None)

    st.divider()


def main():
    st.title('Análisis automático de imágenes retinianas')
    st.markdown(
        'Análisis de angiografías de fluoresceína de modelos murinos para el '
        'estudio de la retinopatía diabética. Trabajo Fin de Máster en '
        'colaboración con el Centro de Investigación Príncipe Felipe (CIPF).'
    )

    ensemble = get_ensemble()

    # Barra lateral
    with st.sidebar:
        st.header('Información')
        st.write(f'**Dispositivo de cómputo:** {"GPU (CUDA)" if USE_GPU else "CPU"}')
        if ensemble is not None:
            st.write('**Clasificador:** disponible')
            st.caption(
                f'Ensemble de {len(ensemble.models)} ResNet18 · '
                f'img_size={ensemble.img_size} · umbral={ensemble.threshold}'
            )
        else:
            st.write('**Clasificador:** no disponible')
            st.caption(f'Origen configurado: `{get_model_url()}`')
        st.divider()
        st.markdown(
            '**Etapas del análisis**\n\n'
            '1. Preprocesado (canal verde, máscara FOV, CLAHE)\n'
            '2. Segmentación vascular (Frangi multiescala y top-hat)\n'
            '3. Métricas vasculares\n'
            '4. Detección de fugas de fluoresceína\n'
            '5. Clasificación Control frente a db/db'
        )

    uploaded_files = st.file_uploader(
        'Seleccione una o varias imágenes de fondo de ojo (.tif)',
        type=['tif', 'tiff'],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info('Suba al menos una imagen .tif para iniciar el análisis.')
        return

    for uploaded_file in uploaded_files:
        with st.spinner(f'Analizando {uploaded_file.name}…'):
            res, clf = analyze_uploaded_file(uploaded_file, ensemble)
        render_result(uploaded_file.name, res, clf, ensemble)


if __name__ == '__main__':
    main()

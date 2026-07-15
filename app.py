"""Demo Streamlit - Análisis automático de imágenes retinianas (TFM / CIPF).

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
import json
import html
import tempfile

import numpy as np
import pandas as pd
import urllib.request
from scipy import ndimage
from skimage.morphology import skeletonize

import streamlit as st
import streamlit.components.v1 as components

from core.device import DEVICE
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
    page_title='Análisis de imágenes retinianas (v2) - TFM/CIPF',
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


# Gradiente de azul por calibre: fino = azul claro, grueso = azul oscuro.
VESSEL_THIN = np.array([150, 200, 255])   # azul claro (vasos finos)
VESSEL_THICK = np.array([0, 40, 150])     # azul oscuro (vasos gruesos)


def caliber_color_map(vessel_mask: np.ndarray, skeleton=None,
                      dist_transform=None) -> np.ndarray:
    """Damos a cada píxel un color RGB según el calibre local del vaso.

    Estimamos el calibre de cada punto como 2x la distancia al borde sobre el eje
    del vaso (el esqueleto) y lo propagamos a toda la sección, de modo que cada
    vaso queda coloreado de forma uniforme según su grosor: los finos en azul
    claro y los gruesos en azul oscuro.
    """
    vm = vessel_mask > 0
    colors = np.zeros((*vm.shape, 3), dtype=np.uint8)
    if not vm.any():
        return colors
    if dist_transform is None:
        dist_transform = ndimage.distance_transform_edt(vm)
    if skeleton is None:
        skeleton = skeletonize(vm)
    if not skeleton.any():
        skeleton = vm
    radius_on_skel = dist_transform * skeleton
    # Cada píxel toma el calibre del punto de esqueleto más cercano.
    _, inds = ndimage.distance_transform_edt(~skeleton.astype(bool), return_indices=True)
    caliber = 2.0 * radius_on_skel[inds[0], inds[1]]
    cv = caliber[vm]
    cmax = float(np.percentile(cv, 95)) if cv.size else 1.0
    cmin = 1.0
    norm = np.clip((caliber - cmin) / max(cmax - cmin, 1e-6), 0.0, 1.0)
    grad = (VESSEL_THIN + norm[..., None] * (VESSEL_THICK - VESSEL_THIN)).astype(np.uint8)
    colors[vm] = grad[vm]
    return colors


def make_vessel_overlay(img_rgb: np.ndarray, vessel_mask: np.ndarray,
                        skeleton=None, dist_transform=None) -> np.ndarray:
    overlay = img_rgb.copy()
    vm = vessel_mask > 0
    colors = caliber_color_map(vessel_mask, skeleton, dist_transform)
    overlay[vm] = colors[vm]
    return overlay


def make_leak_overlay(img_rgb: np.ndarray, vessel_mask: np.ndarray,
                      leakage_mask: np.ndarray,
                      skeleton=None, dist_transform=None) -> np.ndarray:
    overlay = img_rgb.copy()
    vm = vessel_mask > 0
    colors = caliber_color_map(vessel_mask, skeleton, dist_transform)
    overlay[vm] = colors[vm]
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


def render_metrics_table(mdf: pd.DataFrame, key: str):
    """Mostramos la tabla de métricas con un único botón 'Copiar' integrado.

    Al pulsarlo copiamos los datos tabulados al portapapeles, de modo que al
    pegar en Excel o Sheets cada métrica y su valor caen en columnas separadas.
    """
    headers = list(mdf.columns)
    body_rows = mdf.astype(str).values.tolist()
    tsv = mdf.to_csv(sep='\t', index=False)

    thead = ''.join(f'<th>{html.escape(h)}</th>' for h in headers)
    tbody = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in row) + '</tr>'
        for row in body_rows
    )

    html_block = f"""
    <style>
      .mt-wrap {{ font-family: "Source Sans Pro", -apple-system, Segoe UI, Roboto, sans-serif; }}
      .mt-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
      .mt-title {{ font-size:0.95rem; font-weight:600; }}
      .mt-btn {{ border:1px solid #b9c0c8; background:#f6f8fa; color:#24292f;
                 border-radius:6px; padding:4px 12px; font-size:0.82rem; cursor:pointer; }}
      .mt-btn:hover {{ background:#eef1f4; }}
      table.mt {{ border-collapse:collapse; width:100%; font-size:0.9rem; }}
      table.mt th, table.mt td {{ border:1px solid #d9dde1; padding:6px 12px; text-align:left; }}
      table.mt th {{ background:#f2f4f6; font-weight:600; }}
      table.mt td:last-child, table.mt th:last-child {{ text-align:right;
                 font-variant-numeric:tabular-nums; }}
      @media (prefers-color-scheme: dark) {{
        .mt-title {{ color:#e6e6e6; }}
        .mt-btn {{ border-color:#3a3f44; background:#2b2f36; color:#e6e6e6; }}
        .mt-btn:hover {{ background:#343a42; }}
        table.mt th, table.mt td {{ border-color:#3a3f44; color:#e6e6e6; }}
        table.mt th {{ background:#22262c; }}
      }}
    </style>
    <div class="mt-wrap">
      <div class="mt-head">
        <span class="mt-title">Métricas vasculares</span>
        <button class="mt-btn" id="btn-{key}" onclick="copyMetrics_{key}()">Copiar</button>
      </div>
      <table class="mt"><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
    </div>
    <script>
      function copyMetrics_{key}() {{
        const data = {json.dumps(tsv)};
        const ta = document.createElement('textarea');
        ta.value = data;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        let ok = false;
        try {{ ok = document.execCommand('copy'); }} catch (e) {{ ok = false; }}
        if (navigator.clipboard) {{ try {{ navigator.clipboard.writeText(data); ok = true; }} catch (e) {{}} }}
        document.body.removeChild(ta);
        const b = document.getElementById('btn-{key}');
        b.textContent = ok ? 'Copiado' : 'Pulsa Ctrl+C';
        setTimeout(() => {{ b.textContent = 'Copiar'; }}, 1600);
      }}
    </script>
    """
    components.html(html_block, height=90 + 33 * (len(body_rows) + 1))


def analyze_uploaded_file(uploaded_file, ensemble):
    """Guardamos el fichero subido en disco y ejecutamos el pipeline y la clasificación."""
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
    skel = res.get('skeleton')
    dist = res.get('dist_transform')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(res['img_rgb'], caption='Original', use_container_width=True)
    with col2:
        st.image(
            make_vessel_overlay(res['img_rgb'], res['vessel_mask'], skel, dist),
            caption='Red vascular segmentada (calibre: de azul claro en los finos a azul oscuro en los gruesos)',
            use_container_width=True,
        )
    with col3:
        st.image(
            make_leak_overlay(res['img_rgb'], res['vessel_mask'], res['leakage_mask'], skel, dist),
            caption=f'Fugas detectadas (amarillo) - N={res.get("n_leaks", 0)}',
            use_container_width=True,
        )

    # Métricas
    mdf = metrics_dataframe(res)
    table_key = ''.join(ch if ch.isalnum() else '_' for ch in name)
    render_metrics_table(mdf, key=table_key)

    st.divider()


def main():
    st.title('Análisis automático de imágenes retinianas - versión 2')
    st.markdown(
        'Análisis de angiografías de fluoresceína de modelos murinos para el '
        'estudio de la retinopatía diabética. Trabajo Fin de Máster en '
        'colaboración con el Centro de Investigación Príncipe Felipe (CIPF).'
    )

    ensemble = get_ensemble()

    # Barra lateral
    with st.sidebar:
        st.header('Información')
        st.write('**Dispositivo de cómputo:** CPU')
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

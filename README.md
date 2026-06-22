# Análisis automático de imágenes retinianas (TFM / CIPF)

Demo en **Streamlit** del pipeline desarrollado en mi Trabajo Fin de Máster, en
colaboración con el **Centro de Investigación Príncipe Felipe (CIPF)**, para el
análisis automático de imágenes de fondo de ojo (angiografía de fluoresceína) de
modelos murinos en el estudio de la **retinopatía diabética**.

A partir de una imagen `.tif` de fondo de ojo, la demo:

1. **Preprocesa** la imagen: canal verde, máscara FOV (región del ojo) y CLAHE.
2. **Segmenta la red vascular** con un filtro de Frangi multiescala en PyTorch
   (GPU o CPU) combinado con top-hat morfológico.
3. **Extrae métricas vasculares**: densidad, calibre medio, nº de segmentos,
   tortuosidad media y longitud total de la red.
4. **Detecta fugas de fluoresceína** (zonas hiperfluorescentes difusas),
   excluyendo el disco óptico y la zona vascular central densa.
5. **Clasifica** la imagen como **Control** (sano) o **db/db** (enfermo) con un
   ensemble de 5 ResNet18.

> Nota: es una herramienta para validar la efectividad de lo desarrollado, no
> un producto clínico final.

---

## Estructura del proyecto

La lógica del pipeline está **separada de la interfaz**, de modo que los mismos
módulos `core/` se puedan reutilizar desde una futura app de escritorio.

```
streamlit_retina/
├── app.py                  # Interfaz Streamlit
├── core/
│   ├── device.py           # Detección automática GPU/CPU
│   ├── preprocessing.py    # Canal verde, máscara FOV, CLAHE
│   ├── segmentation.py     # Frangi multiescala (PyTorch) + top-hat
│   ├── metrics.py          # Densidad, calibre, segmentos, tortuosidad, longitud
│   ├── leakage.py          # Detección de fugas de fluoresceína
│   ├── classifier.py       # Carga del ensemble e inferencia
│   └── pipeline.py         # Orquestador (preproceso → segmentación → métricas → fugas)
├── models/                 # Aquí va el modelo .pt (no se versiona, ver abajo)
├── sample_images/          # Imágenes de prueba opcionales
├── requirements.txt
└── README.md
```

Las funciones de `core/` reciben imágenes/arrays y devuelven resultados, **sin
visualización**.

---

## Ejecución en local

Requiere Python 3.10+.

```bash
# 1. (Recomendado) crear un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el modelo entrenado en models/
#    models/clasificador_retina_ensemble.pt

# 4. Lanzar la app
streamlit run app.py
```

La app se abre en `http://localhost:8501`. Sube una o varias imágenes `.tif` y
verás, para cada una: la imagen original, la red vascular segmentada, las fugas
detectadas, la tabla de métricas y el veredicto del clasificador.

Funciona automáticamente en **GPU** (si hay CUDA) o en **CPU**. La ruta del
modelo se puede sobreescribir con la variable de entorno `RETINA_MODEL_PATH`.

---

## El modelo (`clasificador_retina_ensemble.pt`)

Es un diccionario de PyTorch con un **ensemble de 5 ResNet18** entrenados con
transfer learning (LOAO-CV, 5 seeds). Estructura esperada:

| clave | contenido |
|---|---|
| `ensemble_state_dicts` | lista de 5 `state_dict` (ResNet18 con `fc` modificada) |
| `arch` | `'resnet18'` |
| `img_size` | tamaño de entrada (224) |
| `threshold` | umbral de decisión (0.5) |
| `seeds` | `[0, 1, 2, 42, 123]` |
| `animals` | IDs de animales de entrenamiento |

**Entrada al modelo (idéntica al notebook):** el canal verde crudo de la imagen,
replicado a 3 canales, redimensionado a 224×224 y normalizado con
`mean=[0.5], std=[0.5]`. Se promedian las probabilidades softmax de los 5
modelos y se decide con el `threshold`. La clase 1 es **db/db**.

---

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) crea una nueva app
   apuntando a `app.py`.
3. Streamlit Cloud instala `requirements.txt` automáticamente.

### Gestión del modelo `.pt` (importante)

El modelo (ensemble de 5 ResNet18) pesa **~214 MB**, por encima del límite de
100 MB para archivos normales de GitHub. Opciones:

- **GitHub Release o Hugging Face Hub (recomendado).** Sube el `.pt` como
  *asset* de una release (o a un repo de modelos en HF). La descarga de assets
  de release es gratuita y **no consume cuota de LFS**, así que es lo más
  robusto para una demo que se rehace varias veces. La app ya soporta esto: si
  el `.pt` no está en disco, lo descarga de la URL indicada en la variable
  `RETINA_MODEL_URL` (variable de entorno o secreto de Streamlit) y lo cachea.

  En Streamlit Cloud: *Settings → Secrets* y añade:
  ```toml
  RETINA_MODEL_URL = "https://github.com/JesusGSoriano/streamlit_retina/releases/download/v1.0/clasificador_retina_ensemble.pt"
  ```

- **Git LFS.** Funciona (cabe en el 1 GB de almacenamiento), pero LFS solo da
  **1 GB de ancho de banda/mes** gratis y Streamlit Cloud descarga el modelo en
  cada *rebuild*; con ~5 rebuilds se agota la cuota y la app deja de poder
  cargar el modelo. Si aun así lo usas: `git lfs install`, `git lfs track
  "*.pt"`, commitear `.gitattributes` y el `.pt`. (Ojo: el `.gitignore` **no**
  debe ignorar `models/*.pt` o LFS no podrá añadirlo.)

> 5 ResNet18 en CPU caben de sobra en la RAM del tier gratuito de Streamlit
> Cloud (~1 GB). El `@st.cache_resource` carga el ensemble **una sola vez** por
> sesión del servidor.

### Wheel de PyTorch en CPU (opcional, despliegue más ligero)

El wheel por defecto de `torch` incluye librerías CUDA y es grande. Para un
despliegue solo-CPU más ligero puedes forzar el wheel de CPU añadiendo al
principio de `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Notas de fidelidad

El pipeline replica **literalmente** las funciones del notebook del TFM
(escalas de Frangi, umbrales de histéresis, parámetros de detección de fugas,
transforms del clasificador, etc.). El objetivo es que los resultados de la demo
sean idénticos a los del notebook.

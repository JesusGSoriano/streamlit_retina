"""Núcleo del pipeline de análisis de imágenes retinianas (TFM / CIPF).

Lógica separada de la interfaz: estos módulos reciben imágenes/arrays y
devuelven resultados, sin visualización. Reutilizables desde Streamlit o desde
una app de escritorio.

Importa desde los submódulos concretos (p.ej. `from core.pipeline import
analyze_image`). El paquete no importa nada de forma eager a propósito, para no
encadenar imports pesados (torch, etc.) bajo el lock de importación, que en
Streamlit Cloud puede provocar bloqueos.
"""

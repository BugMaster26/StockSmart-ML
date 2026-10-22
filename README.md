# StockSmart ML

Aplicación académica en Python y Streamlit para explorar transacciones de ventas, clasificar demanda semanal, agrupar productos y generar alertas orientativas comparando demanda reciente con el stock actual ingresado por el usuario.

> Prototipo académico: la demo utiliza datos sintéticos. El dataset real de Mendeley descrito abajo todavía no está integrado ni incluido en el repositorio. No se deben interpretar las métricas de la demo como resultados del dataset real.

## Problema y alcance

La aplicación centraliza carga/limpieza, resumen de ventas, clasificación de la demanda siguiente en Baja/Media/Alta, segmentación exploratoria y alertas basadas en una media de cuatro semanas. No es un sistema de compras ni un pronóstico de unidades para producción.

### Objetivo general

Preparar transacciones de ventas y ofrecer análisis reproducibles de clasificación semanal, agrupamiento de productos y alertas orientativas de inventario en una interfaz local.

## Funcionalidades y estado de verificación

- Limpieza y reporte de calidad de transacciones.
- Features semanales con rezagos y etiquetas de la semana siguiente.
- Comparación de KNN, Árbol de Decisión y MLP con corte cronológico.
- Clustering de productos con K-Means y Ward, perfiles, silhouette y dendrograma.
- Captura/importación de stock actual y descarga de alertas CSV.
- Navegación lateral Streamlit y exportación CSV.

En la verificación actual con Python 3.12.10, **31 pruebas pasaron**, `pip check` no encontró conflictos y `compileall` terminó sin errores. pytest informó 14 advertencias deprecadas de Matplotlib/pyparsing.

## Tecnologías

Python 3.12, Streamlit, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, openpyxl, xlrd y pytest. Las dependencias directas están fijadas en `requirements.txt` a las versiones probadas con Python 3.12.10. El MLP es `MLPClassifier` de scikit-learn; no se usa TensorFlow/Keras.

## Requisitos e instalación en Windows

- Windows 10/11 de 64 bits.
- Python 3.12 de 64 bits disponible mediante `py -3.12`.
- Internet para descargar paquetes.

Desde CMD en la raíz del proyecto:

```bat
setup_windows.bat
.venv\Scripts\python.exe -m streamlit run app.py
```

El instalador no instala Python. Si falta 3.12, usa [Python 3.12.10 para Windows](https://www.python.org/downloads/release/python-31210/) y vuelve a ejecutar el script. Si `.venv` usa otra versión, el instalador solicita confirmación antes de reemplazar esa carpeta. Más instrucciones: [instalación y solución de problemas](docs/DOCUMENTACION_TECNICA.md#6-instalación-y-ejecución-en-windows).

Instalación manual equivalente (CMD), si estás creando el entorno por primera vez:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

El repositorio está preparado para Community Cloud: la aplicación de entrada es `app.py`, `requirements.txt` está en la raíz y la demostración se genera en memoria, por lo que no hace falta subir un dataset. Las versiones directas fijadas fueron comprobadas en Python 3.12.10 en Windows; no se ha ejecutado un build en Community Cloud ni se ha verificado el despliegue remoto.

1. Inicia sesión en [Streamlit Community Cloud](https://share.streamlit.io/) con GitHub y autoriza el acceso al repositorio `BugMaster26/StockSmart-ML`.
2. Elige **Create app** y selecciona ese repositorio, la rama `main` y el archivo principal `app.py`.
3. En **Advanced settings**, selecciona Python **3.12** si la plataforma ofrece el selector de versión. No se añade `runtime.txt`: la selección se realiza en la configuración de la aplicación y se evita fijar un archivo de runtime antiguo.
4. Pulsa **Deploy**. Espera a que termine la instalación de `requirements.txt` y el arranque. Si falla, abre **Manage app** y consulta los logs antes de cambiar dependencias.
5. Abre la URL `.streamlit.app` que asigne la plataforma y comprueba el panel con **Demostración**. Para probar datos propios, cambia a **Cargar archivo**; el repositorio no trae datos reales.

No se requieren secretos de Streamlit ni variables de entorno para el funcionamiento actual. La aplicación procesa archivos cargados en memoria y ofrece salidas descargables; evita cargar información personal o confidencial en una aplicación pública. El inicio local y las pruebas automatizadas no certifican disponibilidad, rendimiento o compatibilidad en producción; esas condiciones solo pueden confirmarse tras desplegar y probar la URL asignada.

## Dataset

La fuente elegida para la siguiente integración es [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), versión 1, DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1). Mendeley publica los archivos `retail_sales_ml_apl.csv` (37.3 MB) y `retail_inventory_ml_apl.csv` (95.4 MB), y describe una muestra de 40 tiendas, 2,326 SKU y tres divisiones. La ficha indica ventas transaccionales diarias y stock disponible (“On hand”), pero no se han podido verificar las columnas, el periodo de observaciones, la granularidad exacta del inventario ni la relación temporal entre ambos archivos. Por ello, esta versión todavía no integra el dataset real ni genera resultados basados en él.

Para preparar una inspección local, descarga ambos CSV desde **Download All** en la ficha oficial y colócalos en `data/raw/` con esos nombres exactos. Esa carpeta y sus archivos están excluidos de Git; no subas los CSV a GitHub. Tras inspeccionarlos se podrá establecer un mapeo de columnas y validar fechas, claves y reglas de limpieza antes de habilitar su uso. Mientras tanto, la aplicación conserva su modo de demostración y la carga genérica actual, cuyo esquema requerido es `InvoiceDate`, `StockCode`, `Quantity` y `UnitPrice`; no se afirma que los archivos Mendeley tengan esas columnas.

## Estructura

```text
app.py                  Interfaz y orquestación Streamlit
src/                    Limpieza, features, clasificación, clustering y alertas
tests/                  Pruebas unitarias y de interfaz
data/                   README; los datos locales se ignoran en Git
models/                 Marcador de posición; no se guardan modelos entrenados
docs/                   Informe, metodología, manual y guía de exposición
requirements.txt        Dependencias directas fijadas
setup_windows.bat       Configuración guiada en Windows
```

## Algoritmos

- Clasificación: KNN, Árbol de Decisión, MLP.
- Agrupamiento: K-Means y jerárquico Ward sobre variables de producto escaladas.
- Alertas: media de las últimas cuatro semanas calendario comparada con stock actual; no usa los modelos de clasificación.

KNN clasifica por vecinos cercanos; el árbol usa reglas sucesivas; el MLP es una red multicapa. K-Means asigna productos a centroides; Ward combina grupos jerárquicamente. Estos métodos son exploratorios y no han sido validados en una empresa.

## Pruebas

Tras instalar el entorno, ejecutar desde CMD:

```bat
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py src tests
.venv\Scripts\python.exe -m pip check
```

Resultados y limitaciones del entorno usado para auditoría: ver [Documentación técnica](docs/DOCUMENTACION_TECNICA.md#11-pruebas-automatizadas).

## Limitaciones conocidas

El dataset de Mendeley está seleccionado, pero su esquema y periodo aún no se han inspeccionado y no está integrado. No hay ventas reales incluidas, evaluación empresarial, validación temporal repetida ni guardado/carga de modelos. Ventas observadas pueden ocultar demanda durante quiebres. La clasificación usa una sola partición cronológica y clases relativas a cuantiles. La alerta actual no considera estacionalidad, plazos de entrega, compras en tránsito ni stock de seguridad; compara ventas recientes con stock ingresado manualmente. K se escoge manualmente.

## Créditos y licencia

La fuente objetivo para datos es [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), de Jimmy Smith, versión 1 (2026), DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1). La ficha declara CC BY 4.0; si se usan esos datos, se debe conservar la atribución y señalar modificaciones. Los CSV no se incluyen en este repositorio. No se ha definido una licencia para el código y no se incluye `LICENSE`; la visibilidad pública del repositorio no concede por sí sola permisos de copia, modificación o redistribución del código.

## Documentación

- [Datos y pasos para la integración pendiente](data/README.md)
- [Informe académico](docs/INFORME_ACADEMICO.md)
- [Documentación técnica](docs/DOCUMENTACION_TECNICA.md)
- [Metodología de Machine Learning](docs/METODOLOGIA_MACHINE_LEARNING.md)
- [Manual de usuario](docs/MANUAL_USUARIO.md)
- [Guía de exposición (8–10 minutos)](docs/GUIA_EXPOSICION.md)

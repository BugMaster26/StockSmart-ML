# StockSmart ML

Aplicación académica en Python y Streamlit para explorar transacciones de ventas, clasificar demanda semanal, agrupar productos y generar alertas orientativas comparando demanda reciente con el stock actual ingresado por el usuario.

> Prototipo académico: la demo utiliza datos sintéticos. El dataset Online Retail de UCI no está incluido en el repositorio. No se deben interpretar las métricas de la demo como resultados de una empresa o del dataset real.

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

La fuente sugerida es [Online Retail de UCI](https://archive.ics.uci.edu/dataset/352/online+retail), pero el archivo no está incluido. Carga un CSV, XLSX, XLSM o XLS con estas columnas exactas: `InvoiceDate`, `StockCode`, `Quantity`, `UnitPrice`. `InvoiceNo` es opcional para reconocer cancelaciones. La demo es sintética y reproducible.

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

No hay ventas reales incluidas, evaluación empresarial, validación temporal repetida ni guardado/carga de modelos. Ventas observadas pueden ocultar demanda durante quiebres. La clasificación usa una sola partición cronológica y clases relativas a cuantiles. La alerta no considera estacionalidad, plazos de entrega, compras en tránsito ni stock de seguridad. K se escoge manualmente. La tabla `data/README.md` describe incorrectamente `UnitPrice` como opcional; el código la requiere.

## Créditos y licencia

La fuente sugerida para datos es Online Retail, del UCI Machine Learning Repository; el archivo no se distribuye con este proyecto. No se ha definido una licencia para el código y no se incluye `LICENSE`. La publicación permite consultar el repositorio, pero no concede por sí misma permisos de copia, modificación o redistribución. La persona titular debe elegir una licencia antes de otorgar esos permisos.

## Documentación

- [Informe académico](docs/INFORME_ACADEMICO.md)
- [Documentación técnica](docs/DOCUMENTACION_TECNICA.md)
- [Metodología de Machine Learning](docs/METODOLOGIA_MACHINE_LEARNING.md)
- [Manual de usuario](docs/MANUAL_USUARIO.md)
- [Guía de exposición (8–10 minutos)](docs/GUIA_EXPOSICION.md)

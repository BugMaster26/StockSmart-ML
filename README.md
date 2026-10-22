# StockSmart ML

Aplicación académica en Python y Streamlit para explorar transacciones de ventas, clasificar demanda semanal, agrupar productos y generar alertas orientativas comparando demanda reciente con el stock actual ingresado por el usuario.

> Prototipo académico: conserva demo sintética, carga propia y modo integrado con los CSV reales de Mendeley (los originales no se incluyen en Git). Las métricas de esta revisión son una ejecución local de los CSV disponibles, no una validación empresarial.

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

En esta revisión con el entorno `.venv` local, **43 pruebas pasaron** (14 advertencias deprecadas de Matplotlib/pyparsing); `pip check` no encontró conflictos y `compileall` terminó sin errores. La suite incluye comprobaciones con los dos CSV reales disponibles localmente.

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
5. Abre la URL `.streamlit.app` asignada y comprueba el panel con **Demostración**. El modo Mendeley necesita ambos CSV locales o cargados en la interfaz; el repositorio no los incluye.

No se requieren secretos de Streamlit ni variables de entorno para el funcionamiento actual. La aplicación procesa archivos cargados en memoria y ofrece salidas descargables; evita cargar información personal o confidencial en una aplicación pública. El inicio local y las pruebas automatizadas no certifican disponibilidad, rendimiento o compatibilidad en producción; esas condiciones solo pueden confirmarse tras desplegar y probar la URL asignada.

## Dataset real y resultados medidos

La aplicación integra [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), versión 1, DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1), de Jimmy Smith. La ficha describe una muestra de 40 tiendas, 2,326 SKU y tres divisiones. La inspección local encontró 125,751 registros de ventas agregados (118,258 positivos usados para demanda), del 2025-06-01 al 2026-04-24, y 284,755 intervalos de inventario; 22,701 tienen fin abierto (`9999-12-31`). Detalles y limitaciones: [data/README.md](data/README.md).

Ejecución medida con los CSV locales y una división cronológica por fecha (entrenamiento: 54,694 filas hasta 2026-02-09; prueba: 12,697 desde 2026-02-16):

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Árbol de Decisión | 0.504922 | 0.480136 | 0.499123 | 0.479097 |
| KNN | 0.468300 | 0.446650 | 0.462113 | 0.424771 |
| MLP | 0.426636 | 0.420194 | 0.423812 | 0.415436 |

Las clases Baja/Media/Alta se construyen con cuantiles globales de cantidad semanal. Es una comparación exploratoria, no un pronóstico validado ni una validación cruzada temporal. En una corrida de clustering sobre 2,326 perfiles, silhouette fue 0.534953 para K-Means y 0.759648 para Ward con 3 grupos; mide separación geométrica, no valor de negocio.

Hay 7,491 devoluciones con cantidades negativas. Se conservan en calidad y resúmenes netos, y se excluyen de demanda, modelos y alertas. No se especifican moneda ni unidades físicas. El inventario solo se asocia por producto, tienda, canal y fecha cuando existe exactamente un intervalo compatible: 85,461 ventas positivas encontraron uno, 32,797 no y ninguna tuvo varios. El momento de medición de `Qty on hand` no está documentado; las alertas son orientativas y no prueban roturas de stock.

Coloca los originales en `data/raw/retail_sales_ml_apl.csv` y `data/raw/retail_inventory_ml_apl.csv`. Esa carpeta está excluida de Git. Se conserva la demo sintética, la carga propia y la opción Mendeley local o por interfaz. Los archivos grandes se cachean.
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
- Alertas: con Mendeley compara las ventas de cuatro semanas calendario completas con `Qty on hand` de un único intervalo compatible. Los casos sin relación, con cero o con cantidad negativa se señalan sin afirmar rotura de stock. Para archivos propios se conserva el ingreso manual de stock.

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

El conjunto es una muestra estratificada. No se especifican moneda ni unidades físicas, los motivos de devolución están vacíos y la temporización de `Qty on hand` dentro de sus intervalos no está descrita. Hay huecos temporales, cantidades de stock negativas y ventas sin intervalo compatible. Las ventas observadas no identifican demanda no atendida y las alertas no confirman roturas. La clasificación usa cuantiles relativos y una única división cronológica; no hay validación temporal repetida ni evaluación empresarial. No se guardan modelos entrenados. El análisis de clustering es descriptivo, y K se selecciona en la interfaz. Las alertas no consideran plazos de reposición, compras en tránsito, estacionalidad o stock de seguridad.
## Créditos y licencia

La fuente objetivo para datos es [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), de Jimmy Smith, versión 1 (2026), DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1). La ficha declara CC BY 4.0; si se usan esos datos, se debe conservar la atribución y señalar modificaciones. Los CSV no se incluyen en este repositorio. No se ha definido una licencia para el código y no se incluye `LICENSE`; la visibilidad pública del repositorio no concede por sí sola permisos de copia, modificación o redistribución del código.

## Documentación

- [Datos reales: esquema, integración, ejecución y límites](data/README.md)
- [Informe académico](docs/INFORME_ACADEMICO.md)
- [Documentación técnica](docs/DOCUMENTACION_TECNICA.md)
- [Metodología de Machine Learning](docs/METODOLOGIA_MACHINE_LEARNING.md)
- [Manual de usuario](docs/MANUAL_USUARIO.md)
- [Guía de exposición (8–10 minutos)](docs/GUIA_EXPOSICION.md)

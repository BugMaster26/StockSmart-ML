# Guía de exposición de StockSmart ML

## 1. Objetivo de la exposición

Explicar una aplicación académica que organiza transacciones, compara tres clasificadores de demanda semanal, segmenta productos con dos algoritmos y genera alertas simples usando el stock ingresado. La defensa debe separar capacidades implementadas de resultados experimentales: no hay un archivo real en el repositorio, por lo que no se debe presentar la demo sintética como evidencia comercial.

## 2. Orden recomendado de diapositivas (9 minutos)

| Tiempo | Diapositiva | Idea principal |
|---|---|---|
| 0:00–0:45 | 1. Título y objetivo | Qué es StockSmart ML y qué preguntas ayuda a explorar. |
| 0:45–1:35 | 2. Problema y alcance | Calidad de ventas, patrones de demanda, grupos y necesidad de stock actual. |
| 1:35–2:35 | 3. Datos y limpieza | Fuente UCI opcional, esquema mínimo, exclusiones y demo sintética identificada. |
| 2:35–3:45 | 4. Ventanas y fuga | Cómo se predice `t+1` usando semanas observadas hasta `t`. |
| 3:45–5:00 | 5. Clasificadores y evaluación | KNN, árbol, MLP, corte cronológico y métricas disponibles. |
| 5:00–6:10 | 6. Segmentación | Features de producto, escala, K-Means, Ward, perfiles y silhouette. |
| 6:10–7:20 | 7. Alertas | Stock ingresado, media de cuatro semanas, insuficiencia y CSV. |
| 7:20–8:10 | 8. Aplicación y pruebas | Navegación Streamlit, arquitectura, estado de pruebas/entorno. |
| 8:10–9:00 | 9. Limitaciones y cierre | Qué no concluye el prototipo y mejoras necesarias. |

Se puede ampliar a 10 minutos dando más tiempo a la matriz de confusión y preguntas; no conviene cubrir código línea por línea.

## 3. Guion oral sugerido (aprox. 9 minutos)

### 0:00–0:45 | Apertura

“Buenos días. Presento StockSmart ML, una aplicación académica en Python y Streamlit para explorar transacciones de venta. El sistema organiza los datos y ofrece tres tipos de análisis: clasifica la demanda semanal en bandas, agrupa productos por características similares y crea alertas orientativas al comparar una estimación reciente con el stock actual que ingresa el usuario. Su propósito es apoyar la exploración y enseñar un flujo de Machine Learning; no realiza compras ni sustituye un sistema de inventario.”

### 0:45–1:35 | Problema y alcance

“Una hoja de transacciones puede contener cancelaciones, devoluciones, faltantes y semanas sin registros. Además, una venta baja no siempre significa poca demanda: pudo haber falta de existencias. La aplicación separa limpieza, modelos, segmentación y alertas en módulos. En su alcance actual no guarda stock ni se conecta con una tienda. Por eso, las alertas se deben leer como una señal para revisar.”

### 1:35–2:35 | Datos y limpieza

“El origen público sugerido es Online Retail, de UCI, que registra transacciones de una tienda británica. El archivo no está incluido en el proyecto y no he usado sus métricas para afirmar resultados. La aplicación espera InvoiceDate, StockCode, Quantity y UnitPrice; InvoiceNo es opcional para reconocer cancelaciones. La limpieza convierte tipos, informa faltantes y duplicados y excluye registros inválidos, cancelaciones identificables, cantidades no positivas, precios negativos y duplicados exactos. Si selecciono Demostración, la app genera datos sintéticos y lo advierte en pantalla.”

### 2:35–3:45 | Ventanas y fuga de información

“Para clasificación, primero se suma la cantidad por producto y semana. Las semanas de lunes a domingo sin ventas se completan con cero dentro del historial. En una fila de referencia t, lag_1 representa la venta ya observada en t; lag_2 y lag_3 son semanas anteriores. Mean_4 y std_4 resumen cuatro semanas, y trend_4 compara lag_1 con lag_3. La etiqueta es la demanda de t más uno. Por construcción, esa etiqueta se desplaza desde una semana posterior y no se incorpora a los predictores de la fila. Luego se separan semanas únicas cronológicamente; umbrales de clases, imputador, escalador y codificador se ajustan con entrenamiento.”

### 3:45–5:00 | Modelos y métricas

“Se comparan KNN, un Árbol de Decisión y un MLP de scikit-learn. El producto se convierte a variables one-hot y los predictores numéricos se imputan y estandarizan dentro de pipelines. La demanda se convierte a las clases Baja, Media y Alta por cuantiles, que son relativos al entrenamiento. Reportamos accuracy, precision, recall y F1 macro, también por clase, además de la matriz de confusión. El F1 macro da el mismo peso a cada clase. La interfaz resalta al modelo con mayor F1 macro en esa partición, pero eso no demuestra que sea el mejor en otras fechas o empresas. No hay ajuste de hiperparámetros ni backtesting repetido.”

### 5:00–6:10 | Segmentación

“La tarea de clustering no usa etiquetas de demanda. Resume cada producto en unidades, ingresos, pedidos, semanas activas y estadísticas semanales. Estandariza las seis variables y ejecuta K-Means y agrupamiento jerárquico con Ward. El usuario elige K; el sistema no lo optimiza automáticamente. Vemos dispersión de unidades e ingresos, perfiles medios, silhouette y dendrograma. Los números de grupo son etiquetas arbitrarias: los interpreto mirando el perfil, no como un ranking. El dendrograma puede resumirse a treinta ramas para facilitar lectura; el algoritmo asigna todos los productos.”

### 6:10–7:20 | Alertas

“La alerta se calcula por una vía separada de los modelos. El usuario importa o edita StockCode y Stock actual. Los vacíos no se asumen como cero y se rechazan códigos repetidos o valores negativos. La demanda estimada es la media de las últimas cuatro semanas calendario del historial hasta la última semana del archivo, rellenando semanas sin ventas con cero. Solo marca posible insuficiencia cuando la media supera el stock; si falta historial suficiente, informa que no puede estimar. No contempla lead time, órdenes en tránsito, stock de seguridad ni quiebres de inventario, y se puede exportar la tabla a CSV.”

### 7:20–8:10 | Aplicación y estado de verificación

“La interfaz tiene navegación lateral y seis secciones: panel, carga, modelos, clustering, alertas e informe. La lógica está separada en módulos Python y hay pruebas para limpieza, ventanas, modelos, clustering, alertas y recorridos de Streamlit. En la ejecución verificada con Python 3.12.10, las 31 pruebas pasaron; hubo 14 advertencias deprecadas de Matplotlib/pyparsing. El `pip check` también terminó sin dependencias rotas.”

### 8:10–9:00 | Cierre

“En conclusión, el proyecto integra un flujo didáctico de preparación, clasificación, clustering y alertas, y muestra explícitamente cuándo los datos no alcanzan. Su resultado es una herramienta de exploración, no evidencia de efectividad empresarial. Para avanzar hacen falta probar en un entorno limpio Python 3.12, evaluar un archivo real documentado con cortes temporales repetidos y añadir inventario histórico para estimar demanda no observada y reglas reales de reposición. Gracias.”

## 4. Demostración en vivo

1. Antes de presentar, instala las dependencias en Python 3.12 y comprueba que la aplicación abre. No realices la instalación frente al profesor.
2. Abre el panel con **Demostración**; señala la advertencia sintética.
3. En **Carga de datos**, muestra las columnas y la tabla de calidad. Si no tienes un archivo real preparado, explica el formato con una captura o la tabla mínima del manual; no improvises métricas de UCI.
4. Abre **Modelos de demanda**; muestra el corte, las métricas y la matriz. Di que son valores de esa sesión sintética si no se cargó un dataset real.
5. Abre **Clustering**, cambia K una vez y contrasta un perfil K-Means con Ward; interpreta variables, no el significado de una etiqueta numérica.
6. En **Alertas**, completa un ejemplo ficticio de stock y muestra el cálculo y el CSV. Señálalo como ejemplo ilustrativo si no es stock verdadero.
7. Finaliza en **Informe** para resumir límites.

## 5. Explicación sencilla de los algoritmos

- **KNN:** busca observaciones de entrenamiento parecidas y vota por la clase más frecuente entre sus vecinos.
- **Árbol de Decisión:** aplica reglas sucesivas sobre características para llegar a una categoría.
- **MLP:** combina capas de unidades conectadas y ajusta sus pesos para clasificar; aquí es el MLP de scikit-learn.
- **K-Means:** coloca K centros y asigna cada producto al centro más cercano en las variables escaladas.
- **Ward jerárquico:** une productos/grupos paso a paso para limitar el aumento de variación interna y permite ver el orden de fusiones en un dendrograma.

## 6. Ejemplos de negocio (hipotéticos)

- Una tienda pequeña podría revisar si ciertos SKU tienen patrones semanales parecidos y luego diseñar una política común para analizarlos. El clustering por sí solo no identifica una política óptima.
- Un encargado podría comparar una media de ventas reciente con sus existencias reportadas para priorizar revisión de SKU. Para decidir compra necesita incorporar lead time, quiebres y órdenes abiertas.
- Una persona analista podría detectar que una clase Alta se confunde con Media en el tramo de prueba y discutir el costo de cada tipo de error. No se puede asignar costo sin datos de negocio.

Estos casos ilustran posibles usos; no son resultados ni validaciones de StockSmart ML en empresas.

## 7. Preguntas posibles del profesor y respuestas

1. **¿Cuál es el objetivo del sistema?** — Preparar transacciones y ofrecer clasificación de bandas de demanda, segmentación exploratoria y alertas simples de stock.
2. **¿Qué dataset usó realmente?** — El repositorio no contiene el archivo UCI ni otro archivo real. Hay una demo sintética; no debo atribuirle resultados reales.
3. **¿Qué columnas exige la aplicación?** — `InvoiceDate`, `StockCode`, `Quantity` y `UnitPrice`. `InvoiceNo` es opcional.
4. **¿Por qué se eliminan cantidades negativas?** — Se interpretan como devoluciones y se reportan, pero la demanda objetivo se define como ventas positivas. La lógica no las resta de ventas anteriores.
5. **¿Qué predice el MLP?** — La clase baja/media/alta de la demanda semanal siguiente, no una cantidad continua.
6. **¿Cómo define las clases?** — Por dos cuantiles configurables; en la evaluación los cuantiles se calculan con las demandas de entrenamiento.
7. **¿Qué significa lag_1?** — Cantidad vendida en la semana de referencia ya observada, que es la más reciente usada para predecir la semana siguiente.
8. **¿Cómo evita fuga temporal?** — Predictores terminan en t y objetivo está en t+1; se parte por semanas ordenadas y el pipeline aprende imputación, escala y codificación solo en train.
9. **¿La división temporal es validación cruzada?** — No. Es un único holdout cronológico de semanas, 20% por defecto.
10. **¿Cómo interpreta F1 macro?** — Calcula F1 por cada clase y promedia sin ponderar por cuántas observaciones tenga cada clase.
11. **¿Por qué no elegir el modelo por accuracy?** — La app no impone esa decisión; ordena por F1 macro. En clases desbalanceadas, accuracy puede ocultar resultados pobres de una clase. El costo de error debe decidirse con contexto operativo.
12. **¿Qué es la matriz de confusión?** — Tabla de conteos donde las filas son clases reales y las columnas clases predichas, en orden Baja, Media y Alta.
13. **¿El modelo mostrado como seleccionado es el mejor?** — Solo alcanzó mayor F1 macro en el corte de esa ejecución. No hay evidencia de superioridad general.
14. **¿Cómo se tratan los productos nuevos?** — OneHotEncoder ignora categorías de StockCode no vistas en entrenamiento; sus predictores de historial aún se pueden transformar. No hay una estrategia específica validada para cold start.
15. **¿Qué variables usa clustering?** — Unidades, ingresos, pedidos, semanas activas, media semanal y desviación semanal; se estandarizan.
16. **¿Quién selecciona K?** — El usuario mediante slider. Silhouette se reporta después, no hay búsqueda automática del K óptimo.
17. **¿Qué indica silhouette?** — Cohesión y separación relativas según la distancia aplicada. No prueba utilidad económica ni estabilidad futura.
18. **¿Cómo calcula una alerta?** — Compara stock ingresado con media de las últimas cuatro semanas calendario disponibles; alerta solo si la media es mayor.
19. **¿Qué pasa si falta stock?** — La app no supone cero; pide completar. Si falta historial de ventas suficiente, devuelve “Sin datos suficientes”.
20. **¿Se guardan modelos o datos?** — No se guardan modelos entrenados; archivos de ventas se procesan en memoria. Las tablas se guardan solo si el usuario descarga los CSV.

### Preguntas difíciles sobre calidad y validez

- **¿Cómo sabe que la venta refleja demanda real?** — No lo sabe. Sin inventario histórico no se observan quiebres ni ventas perdidas; la demanda puede estar censurada.
- **¿Por qué es problemática una única partición?** — El resultado depende del periodo de prueba, el tamaño y las clases disponibles. Se recomienda backtesting con varios orígenes temporales.
- **¿Cómo eligió umbrales operativos para Baja/Media/Alta?** — No hay umbrales comerciales. Son cuantiles configurables del entrenamiento; se deben acordar con responsables y costos si se aplican en negocio.
- **¿Se compararon contra baseline ingenuo?** — El código compara los tres clasificadores especificados, no incluye predictor baseline de persistencia/clase mayoritaria.
- **¿Cómo demuestra que el MLP converge?** — Se fija un máximo de 1200 iteraciones y tolerancia `1e-3`, pero no se reporta `n_iter_` ni curva de pérdida en la interfaz. Debe inspeccionarse o añadirse en trabajo futuro.
- **¿La alerta optimiza cantidades para el lead time?** — No. Es comparación entre stock y media semanal; no calcula punto de pedido, stock de seguridad, ni costo.
- **¿Los grupos son segmentos estadísticamente estables?** — No se ha evaluado estabilidad con remuestreo o periodos distintos; se muestran como agrupamiento exploratorio.
- **¿Se ejecutó el instalador `.bat`?** — No. La instalación y las pruebas se verificaron directamente con el Python 3.12.10 ya disponible en `.venv`; la secuencia del instalador fue revisada estáticamente.

## 8. Aspectos para estudiar antes de exponer

- Diferencia entre clasificación de clases y predicción numérica de demanda.
- Alineación temporal exacta: features en t y etiqueta en t+1.
- `lag_1`, `lag_2`, `lag_3`, `mean_4`, `std_4` y `trend_4`.
- Qué se calcula solo con entrenamiento: umbrales y transformadores del pipeline.
- Qué significa cada métrica y cómo leer la matriz de confusión.
- Parámetros concretos de KNN, árbol, MLP, K-Means y Ward.
- Qué variables se escalan y diferencia entre asignación y ejes de visualización.
- Regla exacta de las alertas y condiciones de historial insuficiente.
- Diferencia entre datos de demostración y el archivo UCI, que no está incluido.
- Estado honesto de las pruebas y del entorno Python.

## 9. Afirmaciones que deben evitarse sin evidencia

- “La app predice con alta precisión” o “el MLP es el mejor modelo”.
- “Se validó en una empresa” o “redujo faltantes/costos” sin evaluación real.
- “La demo son los datos de Online Retail”.
- “El clustering descubrió segmentos de clientes” (agrupa productos y features concretos).
- “La alerta predice el stock necesario” o “recomienda cuánto comprar”.
- “No existe fuga de información en cualquier escenario”; explicar el diseño implementado y que falta backtesting robusto.
- “Todas las pruebas pasaron” o “Python 3.12 fue verificado” hasta ejecutar la suite completa en una instalación limpia compatible.

## Referencias rápidas

La explicación detallada de datos y algoritmos está en [Metodología de Machine Learning](METODOLOGIA_MACHINE_LEARNING.md). La ficha fuente se encuentra en [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail); los algoritmos están documentados en [scikit-learn](https://scikit-learn.org/stable/user_guide.html).

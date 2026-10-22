# Metodología de Machine Learning de StockSmart ML

## 1. Resumen metodológico

El código implementa dos tareas distintas:

1. **Clasificación supervisada:** a partir del historial reciente de un producto, clasifica la demanda de la semana siguiente como Baja, Media o Alta. Compara KNN, Árbol de Decisión y MLP.
2. **Segmentación no supervisada:** agrupa perfiles agregados de productos con K-Means y clustering aglomerativo Ward.

La alerta de inventario es una tercera rutina determinista: calcula una media móvil reciente y la compara con el stock actual. No usa los clasificadores ni produce un pronóstico numérico con los modelos.

## 2. Fuente y características del dataset

El repositorio no contiene el archivo Online Retail ni otro dataset real. La fuente referenciada es el conjunto [Online Retail de UCI](https://archive.ics.uci.edu/dataset/352/online+retail), que la ficha describe como transacciones de una tienda británica no física entre 2010 y 2011, con 541 909 instancias. El catálogo identifica seis features; su tabla enumera ocho variables: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

| Columna de origen | Interpretación de la ficha UCI | Tratamiento en StockSmart ML |
|---|---|---|
| `InvoiceNo` | Identificador de factura; prefijo C indica cancelación. | Opcional; se usa para excluir cancelaciones y contar pedidos distintos en clustering. |
| `StockCode` | Identificador de producto. | Requerido; se trata como texto y se usa como grupo/categoría. |
| `Description` | Nombre del producto. | Se conserva si existe, no alimenta los modelos. |
| `Quantity` | Unidades transaccionadas. | Requerido; cantidades `<=0` se excluyen de la demanda positiva. |
| `InvoiceDate` | Fecha y hora de la transacción. | Requerida; se convierte a datetime y se agrega por semana. |
| `UnitPrice` | Precio unitario (la ficha indica GBP). | Requerido por la app; precio `<0` se excluye; `Quantity × UnitPrice` crea ingresos. |
| `CustomerID` | Identificador de cliente. | No se usa en los modelos. |
| `Country` | País del cliente. | No se usa en los modelos. |

La fuente y el código no deben confundirse: los campos `StockCode`, `InvoiceDate`, `Quantity` y `UnitPrice` son el esquema mínimo de esta aplicación; no se ha inspeccionado aquí un archivo real para confirmar su contenido, versión, licencia o limpieza.

## 3. Preparación de datos

### 3.1 Carga

`app.py:load_frame` lee CSV, XLSX, XLSM y XLS desde bytes en memoria. `StockCode` se fuerza a texto para preservar, por ejemplo, `0007`. CSV se intenta como UTF-8 con BOM y luego Latin-1 si falla la decodificación. Motores Excel: openpyxl para XLSX/XLSM, xlrd para XLS.

### 3.2 Limpieza

`clean_transactions` requiere cuatro columnas y genera `CleaningReport` con recuentos de origen. Convierte fechas con `pd.to_datetime(errors="coerce")`, y `Quantity`/`UnitPrice` con `pd.to_numeric(errors="coerce")`. Registra faltantes de cada columna, duplicados exactos en entrada, facturas que inician con C y cantidades negativas.

Se excluyen filas que tengan un requerido ausente o inválido, valores no finitos de cantidad/precio, código vacío, cancelación identificable, cantidad no positiva, precio negativo y duplicado exacto. Los códigos supervivientes se recortan y convierten a texto; se calcula `Revenue`. Las devoluciones no se revierten contra una venta, sino que se excluyen del historial de demanda positiva. Un `InvoiceNo` ausente no impide cargar, pero no se pueden detectar cancelaciones por prefijo ni contar pedidos mediante factura.

### 3.3 Agregación semanal y ventanas

Los periodos usan `to_period("W-SUN")` (lunes a domingo) y se representan por el lunes de inicio (`start_time`). La función `build_weekly_features` agrega `Quantity` por producto y semana, rellena ceros entre primera y última semana observada de cada producto, y calcula la fila de referencia `t`:

| Predictor | Fórmula / periodo |
|---|---|
| `lag_1` | `D_t`, demanda de la semana `t` ya observada. |
| `lag_2` | `D_(t-1)`. |
| `lag_3` | `D_(t-2)`. |
| `mean_4` | media móvil de `D_(t-3), …, D_t`. |
| `std_4` | desviación estándar de ventana `D_(t-3), …, D_t`, con el primer valor no definido reemplazado por cero. |
| `trend_4` | `lag_1 - lag_3`. |
| `Demand` (objetivo) | `D_(t+1)`, generado desplazando la serie una semana hacia atrás. |

Las tres primeras posiciones sin las ventanas necesarias se descartan. `StockCode` acompaña a las variables como predictor categórico en los modelos. La prueba `test_features_use_only_observed_weeks_before_target` modifica la última semana objetivo y comprueba que los predictores de la fila que la anticipa no cambien.

### 3.4 Clases de demanda

`build_weekly_features` asigna clases descriptivas por defecto en cuantiles `q=.33` y `q=.67`, ajustables. Baja corresponde a `Demand <= q33`; Media a `q33 < Demand <= q67`; Alta a `Demand > q67`. Estos son cortes relativos al archivo y no límites de negocio.

En `evaluate_models`, los límites se vuelven a calcular **solo usando Demand del tramo de entrenamiento**; luego se asignan las etiquetas a entrenamiento y prueba con esos límites fijos. Esto evita que la distribución futura de etiquetas afecte a los umbrales evaluados. Si el valor alto no es mayor que el bajo, o el entrenamiento queda con una clase, no se genera comparación y se devuelve una advertencia.

La interfaz muestra además los umbrales globales descriptivos que devolvió `build_weekly_features`; esos valores informativos no son los umbrales que se usaron en la evaluación temporal.

### 3.5 Transformación y separación cronológica

La variable `Week` se usa para ordenar y dividir filas por semanas únicas. Con `test_fraction=.2`, `week_cut` es la posición entera `int(n_semanas × .8)`; semanas anteriores a `test_start` son entrenamiento y desde allí, prueba. Así ninguna semana de referencia se reparte entre ambos conjuntos. Como el objetivo de una fila corresponde a la semana siguiente a `Week`, los predictores solo llegan hasta el cierre de su periodo de referencia.

La tabla `ColumnTransformer` tiene dos ramas:

- Numérica: imputación de mediana (`SimpleImputer`) y `StandardScaler` para rezagos/media/desviación/tendencia.
- Categórica: `OneHotEncoder(handle_unknown="ignore")` para `StockCode`.

El preprocesador está dentro de `make_pipeline` junto al clasificador y se ajusta con el tramo train; el test solo recibe la transformación aprendida. Aunque la limpieza y agregación se hacen antes del split, no usan estadísticas globales para transformar los predictores. Evitar esa fuga importa porque estadísticas aprendidas con datos futuros pueden hacer que la evaluación parezca más favorable que un escenario real de predicción.

## 4. Clasificación supervisada

Se entrenan los tres clasificadores cada vez que se solicita la sección (Streamlit cachea el resultado). Todos reciben los mismos seis predictores numéricos y `StockCode`. El tamaño de la prueba se puede pasar como argumento a la función, pero la interfaz no expone ese parámetro; los percentiles sí son ajustables en un slider.

### 4.1 K-Nearest Neighbors (KNN)

1. **Qué es:** clasifica una observación según las etiquetas de los puntos de entrenamiento más cercanos en el espacio transformado.
2. **Uso aquí:** baseline no lineal para asignar clase de demanda de la siguiente semana.
3. **Variables:** los seis predictores temporales escalados y el one-hot de producto.
4. **Ejecución:** se ajusta un `KNeighborsClassifier` dentro del pipeline y predice el tramo posterior.
5. **Hiperparámetros:** `n_neighbors=min(5, len(train))`. `weights`, métrica y algoritmo de búsqueda no se especifican y conservan los defaults de scikit-learn instalado (habitualmente pesos uniformes y distancia Minkowski con `p=2`).
6. **Evaluación:** mismas métricas y matriz de confusión que los otros clasificadores.
7. **Ventajas y límites:** simple y local; depende de escala y de la definición de distancia. La categoría de SKU one-hot participa en la distancia y puede tener influencia. Guardar/comparar todos los puntos puede ser costoso.
8. **Interpretación:** la clase asignada es la clase más votada por vecinos; no es cantidad pronosticada ni probabilidad calibrada que se muestre al usuario.

### 4.2 Árbol de Decisión

1. **Qué es:** divide recursivamente el espacio de features mediante reglas para reducir impureza de las clases.
2. **Uso aquí:** comparar un clasificador basado en particiones de variables temporales y producto.
3. **Variables:** los mismos predictores numéricos y `StockCode` codificado.
4. **Ejecución:** se ajusta en un pipeline. La escala numérica también se ejecuta aunque no suele ser esencial para árboles.
5. **Hiperparámetros:** `max_depth=6`, `class_weight="balanced"`, `random_state=42`; restantes usan defaults de scikit-learn (no se sintonizan en el código).
6. **Evaluación:** accuracy, promedios macro, métricas de cada clase y matriz de confusión en el corte posterior.
7. **Ventajas y límites:** puede representar interacciones con reglas y no requiere que la distancia tenga sentido; un árbol limitado puede perder patrones y sus reglas no se muestran en la interfaz. Un único corte no demuestra estabilidad.
8. **Interpretación:** las predicciones son etiquetas categóricas. La interfaz no muestra reglas ni importancia de features.

### 4.3 Red neuronal MLP

1. **Qué es:** perceptrón multicapa con capas densas y aprendizaje supervisado por optimización iterativa.
2. **Uso aquí:** tercer clasificador para comparar separación no lineal entre Baja/Media/Alta.
3. **Variables:** los mismos seis predictores numéricos transformados y `StockCode` one-hot.
4. **Ejecución:** scikit-learn `MLPClassifier` aprende del conjunto train; no usa TensorFlow ni Keras.
5. **Hiperparámetros:** `hidden_layer_sizes=(32,16)`, `max_iter=1200`, `tol=1e-3`, `early_stopping=False`, `random_state=42`. Activación, solver, regularización y otros parámetros no están configurados y dependen de defaults de la versión instalada.
6. **Evaluación:** las mismas métricas y test temporal que el resto.
7. **Ventajas y límites:** puede modelar relaciones no lineales; tiene menor interpretabilidad, puede no converger antes del límite y no se hace ajuste sistemático de arquitectura, regularización ni semillas.
8. **Interpretación:** devuelve una clase, no una serie continua. El código no muestra probabilidades ni curva de aprendizaje.

## 5. Segmentación no supervisada

### 5.1 Variables de producto

`product_summary` consolida una fila por producto:

- `units`: suma de cantidades positivas.
- `revenue`: suma de `Quantity × UnitPrice`.
- `orders`: número de `InvoiceNo` únicos cuando está disponible; si no, recuento de filas.
- `active_weeks`: semanas con transacciones positivas.
- `weekly_mean`, `weekly_std`: media y desviación estándar de los totales semanales observados; la desviación no definida se sustituye por cero.

Los estadísticos semanales se calculan sobre semanas con transacciones del producto, no sobre una rejilla completa con ceros. Las seis columnas se limpian de infinito/NaN y luego se estandarizan con `StandardScaler`.

### 5.2 K-Means

1. **Qué es:** asigna puntos a centroides y actualiza centroides para minimizar suma de cuadrados intragrupo.
2. **Uso aquí:** segmentar productos con magnitudes, ingresos, pedidos, actividad y variación comparables tras escalar.
3. **Variables:** las seis features estandarizadas anteriores.
4. **Ejecución:** `fit_predict` en todos los productos.
5. **Hiperparámetros:** `n_clusters` seleccionado manualmente, `n_init=10`, `random_state=42`; los demás defaults de scikit-learn.
6. **Evaluación:** silhouette sobre los mismos datos escalados; perfiles de medias se calculan en las unidades originales.
7. **Ventajas y límites:** ejecución generalmente eficiente e interpretación por centroides/perfiles; requiere elegir K, es sensible a escala/outliers y favorece grupos compactos. El estándar reduce efecto de unidades frente a ingresos, pero no elimina asimetrías o valores extremos.
8. **Interpretación:** el entero del grupo es un identificador arbitrario. Se interpreta comparando medias de perfil, no como categoría ordenada.

### 5.3 Clustering jerárquico Ward

1. **Qué es:** agrupamiento aglomerativo que empieza con cada punto individual y une grupos, usando Ward para minimizar incremento de variación intragrupo.
2. **Uso aquí:** contraste jerárquico de la partición K-Means y visualización de relaciones entre perfiles.
3. **Variables:** las mismas seis variables escaladas.
4. **Ejecución:** `AgglomerativeClustering(n_clusters=K, linkage="ward", compute_distances=True)` ajustado sobre todos los productos; se reconstruye formato linkage SciPy para dendrograma.
5. **Hiperparámetros:** K seleccionado por el usuario, Ward y `compute_distances=True`; la métrica de Ward es euclidiana según requisitos del algoritmo. No hay restricciones de conectividad.
6. **Evaluación:** silhouette sobre matriz escalada; perfil medio por etiqueta; dendrograma de distancias de enlace.
7. **Ventajas y límites:** muestra jerarquía de fusiones; el ajuste puede consumir mucha memoria/tiempo en catálogos grandes. No se elige K por el dendrograma automáticamente.
8. **Interpretación:** alturas mayores representan fusiones a mayor distancia. Si hay más de 30 productos, el gráfico se trunca a 30 ramas finales; etiquetas/grupos calculados siguen considerando todos los productos.

### 5.4 Elección de número de grupos

La interfaz permite K desde 2 hasta `min(8, n_productos-1)` y requiere al menos tres productos. La función rechaza K no entero, menor que 2 o igual/superior al número de productos, además de perfiles escalados distintos insuficientes. Silhouette se presenta para el K escogido y no constituye una búsqueda/selección automática del número de grupos. El ajuste puede producir grupos con tamaños desiguales y el número de cluster no tiene identidad semántica.

## 6. Métricas de evaluación

### Clasificación

- **Accuracy:** fracción de filas de prueba cuya clase predicha coincide con la observada.
- **Precision por clase:** de las filas predichas de esa clase, cuántas corresponden a esa clase real.
- **Recall por clase:** de las filas reales de la clase, cuántas se identificaron.
- **F1 por clase:** media armónica de precision y recall.
- **Macro:** media aritmética sobre las clases Baja, Media y Alta, sin ponderar por soporte. `zero_division=0`; incluye los tres labels incluso si alguno no se predice.
- **Matriz de confusión:** filas = clase real, columnas = clase predicha; orden fijo Baja, Media, Alta.

La app ordena métricas por F1 macro y muestra el primer modelo como “mayor F1 macro en esta partición”. No se debe afirmar que sea universalmente superior. Una clase con menos de diez observaciones de entrenamiento provoca una advertencia; no se corrige automáticamente el desbalance.

### Clustering

Silhouette se calcula solo si las etiquetas contienen más de un grupo y menos grupos que observaciones. Un valor más alto suele indicar mayor separación/cohesión bajo la métrica usada, pero no valida utilidad empresarial, estabilidad temporal ni relación causal.

## 7. Control de fuga de información

El diseño evita tres rutas importantes de fuga:

1. La etiqueta es `D_(t+1)` y los predictores usan solo `D_t` y semanas anteriores.
2. Train/test se separa en límites de semanas únicas ordenadas; no hay filas de la misma semana de referencia en ambos tramos.
3. Cuantiles de clases, imputador, escalador y codificador se ajustan solo en entrenamiento (umbrales de clase por separado; transformadores dentro del pipeline).

El objetivo futuro de una fila de entrenamiento puede caer en la semana que coincide con la fecha de referencia de una fila posterior; esto es propio de una predicción secuencial walk-forward con observaciones previas y no implica que el predictor de esa fila use el futuro. Para una evaluación de producción más estricta conviene definir con precisión la fecha disponible al pronosticar, establecer un gap temporal si corresponde al proceso real y evaluar varios orígenes temporales. El proyecto actual no implementa esos escenarios múltiples.

## 8. Alertas: estimador separado de los modelos

`create_inventory_alerts` valida el stock actual: columnas requeridas, códigos no vacíos y únicos, stock numérico finito y no negativo. Después agrega las ventas positivas por semana de lunes a domingo. Para cada producto crea una rejilla de semanas desde su primera semana hasta la última semana del archivo completo; semanas sin venta se completan con cero. Si hay al menos cuatro semanas en esa rejilla, se calcula la media de las últimas cuatro. Un producto sin suficiente historial o sin correspondencia produce `Sin datos suficientes`. La alerta `Posible insuficiencia` se emite únicamente si estimación > stock; igualdad queda como `Stock suficiente`.

Este cálculo no invoca KNN, árbol ni MLP. Una semana de ventas ausentes porque el producto no tuvo stock se interpreta como cero observado. Para un producto que dejó de vender antes del final global, los ceros del tramo final se incluyen en las últimas cuatro semanas. No se incorpora el stock al historial, no se calcula punto de reorden y no hay plazo de entrega.

## 9. Pruebas y referencias metodológicas

Los tests actuales incluyen:

- `test_preprocessing.py`: duplicados, cancelaciones, valores negativos, requeridos y no finitos.
- `test_features.py`: generación de rezagos, clase, no fuga en objetivo y alineación de lag.
- `test_classification.py`: tres estimadores, orden temporal, límites de train y explicación de historial insuficiente.
- `test_clustering.py`: variables/perfiles, escala, silhouette, linkage/dendrograma, K inválido y perfiles duplicados.
- `test_inventory_alerts.py`: media de cuatro semanas, ceros calendario, historial corto, entradas inválidas y CSV UTF-8.
- `test_app.py`: demo, navegación, alertas y carga CSV/XLSX.

Las versiones declaradas y el resultado de verificación del entorno se documentan en [Documentación técnica](DOCUMENTACION_TECNICA.md); no se incluyen valores de métricas experimentales porque no hay un dataset real cargado en el repositorio.

## Referencias técnicas

- Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*. Journal of Database Marketing & Customer Strategy Management, 19(3). Ficha bibliográfica en [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).
- Pandas Development Team. (s. f.). [Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html).
- Scikit-learn developers. (s. f.). [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [preprocesamiento](https://scikit-learn.org/stable/modules/preprocessing.html) y [métricas de evaluación](https://scikit-learn.org/stable/modules/model_evaluation.html).
- UCI Machine Learning Repository. (2015). [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) [Conjunto de datos].

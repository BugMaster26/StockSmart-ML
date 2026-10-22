# Manual de usuario de StockSmart ML

## 1. Antes de empezar

Instala Python 3.12 de 64 bits y ejecuta `setup_windows.bat` desde CMD en la carpeta del proyecto. Si termina sin errores, inicia la aplicación:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

Abre en el navegador la dirección que presenta Streamlit (normalmente `http://localhost:8501`). Para cerrar el servidor, vuelve a CMD y pulsa `Ctrl+C`. La guía completa de instalación y los problemas de entorno están en [README](../README.md) y [Documentación técnica](DOCUMENTACION_TECNICA.md).

> La suite fue ejecutada con Python 3.12.10: 31 pruebas pasaron. El instalador `.bat` está descrito en la guía, aunque no se ejecutó durante esta auditoría; la instalación activa se validó usando directamente el Python del entorno virtual.

## 2. Navegación

En el panel lateral:

1. En **Origen**, escoge **Demostración** o **Cargar archivo**.
2. En **Navegación**, abre **Panel principal**, **Carga de datos**, **Modelos de demanda**, **Clustering**, **Alertas de inventario** o **Informe**.

La app reejecuta el script al cambiar controles. Se cachean operaciones de lectura/limpieza, features, evaluación y segmentación dentro de la sesión para evitar recomputar los mismos argumentos.

## 3. Elegir y cargar datos

### Demostración

Selecciona **Demostración** para ver datos sintéticos reproducibles generados en memoria. La pantalla muestra la advertencia DEMOSTRACIÓN. Úsala para aprender la navegación, no para afirmar que sus cifras son resultados de Online Retail o de una empresa.

### Archivo de ventas

1. Selecciona **Cargar archivo** en Origen.
2. Usa el cargador que aparece debajo y selecciona un CSV, XLSX, XLSM o XLS.
3. Comprueba que tenga estos nombres exactos:

   ```text
   InvoiceDate, StockCode, Quantity, UnitPrice
   ```

4. Si está disponible, conserva `InvoiceNo`; se usa para reconocer facturas canceladas que empiezan por C. Las demás columnas, como Description, CustomerID y Country, pueden estar presentes, pero no entran en los modelos.
5. Si la carga falla, revisa el mensaje, extensión y columnas. StockCode se importa como texto, lo que conserva códigos con ceros iniciales.

Para CSV, el cargador intenta UTF-8 con BOM y luego Latin-1 cuando falla la decodificación. Para XLSX/XLSM se usa openpyxl; para XLS, xlrd. No hay mapeo automático de nombres como `Invoice Date`, ni selección de hoja desde la interfaz.

**Ejemplo ilustrativo mínimo** (no es un archivo de resultados):

```csv
InvoiceNo,InvoiceDate,StockCode,Quantity,UnitPrice
100001,2024-01-01 10:00:00,0007,3,2.50
100002,2024-01-08 11:00:00,0007,1,2.50
```

La carga se procesa en memoria y no se guarda de forma automática en el repositorio.

## 4. Panel principal

El panel resume el origen seleccionado, ingresos totales, unidades vendidas, productos distintos y filas válidas. El gráfico muestra ingresos agrupados por semana en la moneda consignada en el archivo. La app no convierte ni detecta moneda. Debajo aparecen filas iniciales/válidas y recuentos de duplicados, cancelaciones y cantidades negativas.

Estas cifras dependen de la limpieza aplicada. Una fila descartada no suma al panel.

## 5. Revisar la carga y la calidad

Abre **Carga de datos** para ver las primeras 30 filas originales y el informe de calidad. La tabla incluye faltantes por columna, duplicados, cancelaciones, cantidades negativas y filas inválidas. Los campos requeridos ausentes o no convertibles, valores no finitos, códigos vacíos, cancelaciones, cantidades no positivas, precios negativos y duplicados se excluyen del DataFrame analítico.

La sección permite descargar `ventas_procesadas.csv`. El botón descarga los datos limpios de esa sesión; no escribe un archivo en `data/` automáticamente.

## 6. Entrenar y comparar clasificadores

1. Abre **Modelos de demanda**.
2. Ajusta el slider **Percentiles para separar clases baja, media y alta**. El valor inicial es 33 y 67.
3. Espera a que se calculen KNN, Árbol de Decisión y MLP.
4. Revisa la tabla de accuracy, precision, recall y F1 macro, junto con los valores por clase.
5. El resultado “Mayor F1 macro en esta partición” solo identifica al líder de esa evaluación concreta.
6. Selecciona un modelo en el control **Matriz de confusión**. Las filas son la clase real y las columnas la predicha; el orden es Baja, Media, Alta.
7. Descarga `metricas_modelos.csv` si deseas conservar esa tabla.

Los ejemplos de entrada son la demanda semanal del producto en `lag_1`, `lag_2`, `lag_3`, `mean_4`, `std_4`, `trend_4` y el identificador StockCode. El objetivo es la clase de la semana siguiente, no un número de unidades. El split reserva semanas cronológicamente posteriores; el entrenamiento usa cuantiles calculados solo sobre la parte de entrenamiento. La pantalla muestra fechas de referencia y cantidades de filas.

Si aparece “Datos insuficientes”, hace falta más historial semanal/variación o más de una clase en entrenamiento. Una advertencia de clases con menos de diez ejemplos de entrenamiento indica evaluación frágil; no implica que la app haya reparado el desbalance.

## 7. Leer los gráficos de clustering

1. Abre **Clustering** y selecciona K con el slider (entre 2 y el límite que permita la cantidad de productos; la app exige al menos tres productos).
2. En la pestaña K-Means examina el diagrama de puntos, perfil medio y silhouette.
3. En **Clustering jerárquico** revisa su gráfico y perfil; más abajo está el dendrograma Ward.
4. Consulta la asignación final por producto.

La asignación usa seis variables estandarizadas; los ejes de los gráficos muestran unidades e ingresos originales para facilitar lectura. Los grupos 0, 1, 2, etc. son etiquetas arbitrarias, no ranking. Silhouette resume separación/cohesión bajo una métrica geométrica, pero no demuestra que un grupo sea rentable ni que una política comercial sea adecuada.

Si el catálogo supera 30 productos, el dendrograma se resume visualmente a 30 ramas; las asignaciones usan todos los productos. No hay botón de exportación específico para la tabla de clustering.

## 8. Registrar stock y calcular alertas

1. Abre **Alertas de inventario**.
2. Descarga **plantilla de stock** si te conviene. Tiene `StockCode` y `Stock`; el stock se deja en blanco deliberadamente.
3. Opcionalmente importa un CSV con exactamente `StockCode,Stock`, o introduce/corrige valores en el editor.
4. Completa todos los códigos y cantidades. Los campos vacíos no se consideran cero. Los códigos repetidos, stock negativo, NaN o infinito se rechazan.
5. Revisa demanda semanal estimada y estado. Descarga `alertas_inventario.csv` para conservar la tabla.

**Ejemplo ilustrativo de entrada de stock** (cantidades ficticias):

```csv
StockCode,Stock
0007,20
SKU-02,4
```

La estimación es la media de las últimas cuatro semanas calendario de cada producto hasta la última semana observada en el archivo. Las semanas sin ventas se completan con cero. “Posible insuficiencia” significa estrictamente que la media estimada supera el stock actual; igualdad se clasifica como “Stock suficiente”. “Sin datos suficientes” aparece si no hay cuatro semanas de historial o el código no tiene historial. No se consulta el clasificador.

Las alertas no incluyen proveedor, lead time, stock de seguridad, ventas perdidas, compras en tránsito ni inventario reservado. Confirma los datos y aplica criterio operativo antes de tomar una decisión.

## 9. Informe y exportaciones

La vista **Informe** muestra el origen y el tamaño del conjunto procesado y, solo si se evaluaron modelos para el mismo dataset en la sesión, las métricas y descarga correspondiente. Si no, indica que primero hay que visitar Modelos de demanda.

Exportaciones disponibles:

- Carga de datos: `ventas_procesadas.csv`.
- Modelos e Informe: `metricas_modelos.csv`.
- Alertas: `plantilla_stock_actual.csv` y `alertas_inventario.csv`.

Las descargas CSV usan UTF-8 con BOM para facilitar apertura en hojas de cálculo. El resultado se conserva donde el usuario lo descargue; la app no mantiene historial entre sesiones.

## 10. Errores y datos insuficientes

- **Columnas faltantes:** compara encabezados exactos con las cuatro columnas requeridas.
- **CSV ilegible:** prueba codificación UTF-8; verifica separadores, comillas, fechas y columnas.
- **XLS no abre:** confirma que xlrd se instaló en `.venv`.
- **No hay filas válidas:** revisa fechas, cantidades, precios y códigos vacíos; se excluyen transacciones no positivas.
- **Modelos no comparables:** revisa cuántas semanas únicas y cuántas clases quedan en entrenamiento.
- **Clustering no disponible:** necesita tres o más productos y suficiente cantidad de perfiles distintos.
- **Alertas incompletas:** completa todos los stocks y verifica que el código coincida carácter por carácter.
- **`pip check` o pruebas fallan:** revisa la versión del entorno y ejecuta comandos con `.venv\Scripts\python.exe`, según la guía de instalación. Las dependencias directas están fijadas a las versiones que pasaron la suite con Python 3.12.10.

## 11. Interpretación responsable

Los valores calculados son descripciones y señales para revisión. La demo es sintética; el dataset real no viene incluido. El clasificador predice bandas relativas y usa un único corte temporal. Las ventas observadas pueden no representar demanda cuando hubo quiebres. Las alertas son una media sencilla y el sistema no compra ni recomienda automáticamente cantidades.

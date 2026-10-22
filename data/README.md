# Datos

## Dataset real seleccionado: integración pendiente

La fuente objetivo es [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), versión 1, DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1). La ficha de Mendeley, publicada el 27 de abril de 2026, enumera estos archivos:

| Archivo publicado | Tamaño indicado por Mendeley |
|---|---:|
| `retail_sales_ml_apl.csv` | 37.3 MB |
| `retail_inventory_ml_apl.csv` | 95.4 MB |

La descripción de la ficha habla de ventas diarias a nivel de transacción y stock “On hand”, con una muestra de 40 tiendas, 2,326 SKU y tres divisiones. Estas son afirmaciones de los metadatos; las columnas, el periodo real de las filas, las unidades, las claves y la granularidad/temporización del inventario todavía no se han podido inspeccionar. No hay un mapeo implementado y no debe inferirse que el dataset coincide con las columnas que usa hoy la aplicación.

### Descarga e inspección local

1. Abre la [ficha oficial de Mendeley Data](https://data.mendeley.com/datasets/27x8mjm8k4/1) y pulsa **Download All**.
2. Extrae los archivos descargados y coloca los dos CSV, sin renombrarlos ni editarlos, en `data/raw/`:

   ```text
   data/raw/retail_sales_ml_apl.csv
   data/raw/retail_inventory_ml_apl.csv
   ```

3. Conserva los originales localmente. `.gitignore` excluye `data/`, excepto este README; no publiques estos archivos en GitHub.
4. Con ambos archivos disponibles se podrán inspeccionar columnas, fechas, faltantes, duplicados, unidades y relaciones/granularidad antes de definir la integración.

Los CSV originales no están disponibles en la copia local. Por eso no se ha integrado ni probado el dataset real, no se reportan métricas sobre él y la aplicación no lo carga automáticamente.

### Estado actual de la aplicación

La interfaz conserva los modos **Demostración** y **Cargar archivo**. La demostración es sintética. La carga actual espera `InvoiceDate`, `StockCode`, `Quantity` y `UnitPrice`; `InvoiceNo` es opcional. Las alertas usan stock ingresado/importado manualmente y no unen el inventario Mendeley con las ventas. Hasta revisar los CSV, el dataset real no es una fuente operativa en StockSmart ML.

### Licencia y atribución

La ficha declara [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Si se usan los datos, conserva la atribución a Jimmy Smith, cita el título, versión 1, año 2026 y DOI `10.17632/27x8mjm8k4.1`, y menciona cualquier modificación. No se redistribuyen los archivos originales en este repositorio.

## Demostración

La aplicación genera la demo en memoria y la identifica como **DEMOSTRACIÓN (datos sintéticos)**. No representa observaciones de Mendeley ni resultados de un negocio real.

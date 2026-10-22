# Datos

## Dataset real de Mendeley

Fuente: [Retail Transactions and Stocks Data](https://data.mendeley.com/datasets/27x8mjm8k4/1), versión 1, DOI [10.17632/27x8mjm8k4.1](https://doi.org/10.17632/27x8mjm8k4.1), de Jimmy Smith. La ficha describe una muestra estratificada de 40 tiendas, 2,326 SKU y tres divisiones; enumera dos archivos CSV y declara licencia CC BY 4.0. La inspección local de los archivos descargados encontró:

| Archivo | Filas | Columnas | Periodo observado | Tamaño local aproximado |
|---|---:|---:|---|---:|
| `retail_sales_ml_apl.csv` | 125,751 filas de origen; 118,258 registros positivos usados como ventas | 17 | 2025-06-01 a 2026-04-24 | 37.3 MiB |
| `retail_inventory_ml_apl.csv` | 284,755 intervalos | 18 | inicio: 2025-06-01 a 2026-04-24; fin regular: 2025-06-01 a 2026-04-23 | 95.4 MiB |

El inventario incluye 22,701 filas con `End Date = 9999-12-31`, que se tratan como intervalos abiertos vigentes, no como fechas reales. La ficha habla de ventas diarias y stock “On hand”; las filas observadas tienen claves agregadas por fecha/producto/tienda/canal/tipo y los intervalos del inventario no documentan el momento exacto en que se midió `Qty on hand`.

### Esquema detectado en los archivos

Las fechas de origen se leyeron como texto y se normalizaron a fecha; los tipos siguientes son los inferidos al leer el CSV con pandas. En ventas, el candidato de granularidad `[Transaction Date, Product No, Store, Sales Channel, Sales Type, Is Return]` no tiene claves repetidas. Al omitir `Is Return` aparecen 1,823 combinaciones repetidas que corresponden a registros de venta/devolución, no a duplicados exactos. No hay filas duplicadas exactas en ninguno de los dos archivos.

| Archivo | Columnas de origen y tipos observados |
|---|---|
| Ventas | `Transaction Date` object/fecha; `Sales Type` object; `Is Return` int64; `Reason of Return` float64 (100% faltante); `Supplier`, `Product No`, `Product Description`, `Product Division`, `Product Category`, `Product Subcategory`, `Product Segment`, `Store`, `Sales Channel` object; `Qty Sold`, `Sales Amount`, `Cogs` float64; `Number of Transactions` int64. |
| Inventario | `Start Date`, `End Date` object/fecha; `Stock Status`, `Supplier`, `Product No`, `Product Description`, `Product Division`, `Product Category`, `Product Subcategory`, `Product Segment`, `Store`, `Store Type`, `Sales Channel` object; `Qty on hand`, `Stocks Selling Amount`, `Cost of Stocks`, `Stock Unit Selling Price`, `Stock Unit Cost Price` numéricas (float64). |

El inventario no tiene valores faltantes en las 18 columnas. Su clave candidata de intervalo `[Start Date, End Date, Product No, Store, Sales Channel, Stock Status]` es única; las filas no tienen solapamientos temporales por producto/tienda/canal tras interpretar el centinela como abierto. Hay huecos entre intervalos. Para la ingesta, las columnas necesarias para asociar y comparar demanda son fecha, producto, tienda, canal, cantidad y límites de intervalo; las columnas descriptivas y de importe no se usan como claves.

### Columnas y unidades

Ventas: `Transaction Date`, `Sales Type`, `Is Return`, `Reason of Return`, proveedor, producto y jerarquía del producto, `Store`, `Sales Channel`, `Qty Sold`, `Sales Amount`, `Cogs` y `Number of Transactions`. `Reason of Return` está vacío en las 125,751 filas. `Qty Sold` contiene cantidades negativas en 7,491 filas de devolución, además de dos ceros; la integración conserva las devoluciones en el perfil de calidad y excluye devoluciones/cantidades no positivas de las ventas usadas para demanda, clasificación, clustering, alertas y ventas netas por separado. Para el resumen monetario se mantienen ventas positivas y devoluciones negativas; la moneda no está indicada por los archivos/ficha. La unidad física de cantidad tampoco está especificada.

Inventario: incluye `Start Date`, `End Date`, `Stock Status`, las claves y jerarquía de producto, tienda/canal, `Qty on hand`, valores de venta/costo y precios unitarios. `Qty on hand` presenta 13,869 ceros y 1,380 valores negativos en el archivo; se conservan y se marcan para revisión, no se corrigen a cero. No se especifica unidad física. `Stocks Selling Amount` concuerda con cantidad por precio unitario con una tolerancia de 0.01; `Cost of Stocks` no siempre cuadra con cantidad por costo unitario y no se usa para inferir demanda.

### Relación y compatibilidad temporal

Ambos archivos comparten los mismos 2,326 códigos de producto y 40 tiendas, y tienen un único canal (`Channel Alpha`). La clave analítica de ventas es `Transaction Date` + `Product No` + `Store` + `Sales Channel` + `Sales Type` + `Is Return`; no hay filas duplicadas exactas. Los intervalos de inventario son únicos por `Product No` + `Store` + `Sales Channel` + `Start Date` + `End Date` + `Stock Status`; tampoco hay duplicados exactos. `Product Description` contiene identificadores que no coinciden con `Product No`, por lo que no se usa como clave.

La integración no hace un merge por producto solamente. Busca para cada venta el intervalo inclusivo compatible con la misma fecha, producto, tienda y canal; el estado de ventas/inventario no se usa como clave. En las 118,258 ventas positivas, 85,461 encontraron exactamente un intervalo, 32,797 no encontraron uno y ninguna encontró varios. Hay 16,170 ventas emparejadas cuyo `Sales Type` difiere de `Stock Status`; por esto el estado se presenta como discrepancia descriptiva. Las alertas históricas se limitan a registros con intervalo compatible único, calculan ventas observadas en las cuatro semanas calendario completas anteriores y muestran `Qty on hand` como existencia observada en el intervalo. La fuente no establece si esa cantidad representa el inicio, fin o todo el intervalo; la comparación es orientativa y no prueba rotura de stock.

### Descarga, ubicación y privacidad

Descarga ambos archivos desde **Download All** en la [ficha oficial](https://data.mendeley.com/datasets/27x8mjm8k4/1) y colócalos sin editar en:

```text
data/raw/retail_sales_ml_apl.csv
data/raw/retail_inventory_ml_apl.csv
```

`data/raw/` está excluida de Git. La aplicación permite elegir **Dataset Mendeley** y lee esos archivos locales; si no están ambos, permite cargarlos por los controles de Streamlit. La carga local está cacheada con tamaño y fecha de modificación; una carga subida se cachea por contenido. Para refrescar una copia local reemplaza el CSV y reinicia/limpia la caché de Streamlit si la fecha o el tamaño no cambian.

La página **Calidad de datos** muestra diagnóstico de columnas, tipos, faltantes, duplicados, periodo y cobertura de la relación. Los originales no se modifican. Se pueden descargar las tablas normalizadas desde la interfaz. En Git solo se versiona este archivo de instrucciones, nunca los CSV originales.

### Atribución y límites

La ficha declara [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Al reutilizar los datos, atribuye a Jimmy Smith, incluye el título, versión 1, año 2026 y DOI, e indica las transformaciones descritas aquí. No se redistribuyen los CSV en el repositorio. Las devoluciones carecen de motivo; faltan unidades/currency; el conjunto es una muestra estratificada, no el universo de la cadena; existen huecos temporales; no hay certeza sobre el instante de medición del inventario; y las ventas registradas no revelan demanda no atendida.

## Fuentes conservadas

- **Demostración:** datos sintéticos en memoria, disponibles sin descargar Mendeley.
- **Archivo propio:** carga del esquema genérico documentado en el manual de usuario.
- **Dataset Mendeley:** integración local descrita arriba.

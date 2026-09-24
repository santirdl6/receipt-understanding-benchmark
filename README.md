# Receipt Understanding Benchmark

Banco de pruebas reproducible para extracción de información en tickets. El
proyecto utilizará **CORD** como dataset público y un dataset propio de tickets
españoles, manteniendo un esquema canónico común para todos los modelos.

## Orden experimental

1. CORD con su OCR ground truth + reglas (mide solo la extracción).
2. Imágenes de CORD + PaddleOCR + las mismas reglas (sistema completo).
3. LayoutLMv3.
4. Donut.
5. VLM generalista.

## Estado de esta versión

- Esquema canónico versionado.
- Mapeo de etiquetas CORD al esquema común.
- Adaptador para anotaciones CORD v1/v2 (`valid_line`).
- Conservación de texto, polígonos, cajas, grupos y orden de lectura.
- CLI para inspeccionar y convertir anotaciones.
- Validaciones y pruebas unitarias con una muestra sintética.

No incluye datos de CORD ni pesos de modelos. Los datasets y resultados deben
guardarse fuera del repositorio.

## Instalación local

```bash
git clone <URL_DEL_REPOSITORIO>
cd receipt-understanding-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

## Directorio de datos

En el servidor de la UA, guarda todo lo pesado en `~/datos`:

```bash
export RECEIPT_DATA_ROOT="$HOME/datos/receipt-benchmark"
mkdir -p "$RECEIPT_DATA_ROOT"/{datasets,models,checkpoints,predictions,reports}
```

La variable evita codificar rutas personales en el proyecto.

## Convertir una anotación de CORD

CORD puede presentarse como JSON directo o mediante un registro de Hugging Face
con `ground_truth` serializado como texto. Ambos formatos son aceptados.

```bash
receipt-benchmark inspect-cord /ruta/a/annotation.json
receipt-benchmark convert-cord /ruta/a/annotation.json /ruta/a/canonical.json \
  --document-id cord_train_000001 \
  --image-path /ruta/a/image.png
```

El adaptador no inventa información. Las categorías de CORD sin equivalencia
en el esquema se conservan en `source_label`, pero su `canonical_label` queda
vacía. Los campos españoles que CORD no anota no se consideran negativos.

## Estructura

```text
configs/                 Mapeos declarativos
src/receipt_benchmark/
  adapters/              Dataset original -> esquema canónico
  schema/                Modelos y validaciones comunes
  cli.py                  Comandos de inspección y conversión
tests/                    Pruebas sin descargar datasets
```

## Próximos hitos

- Validar el adaptador con varios ejemplos reales de CORD v2.
- Añadir el baseline de reglas sobre OCR ground truth.
- Integrar PaddleOCR manteniendo las mismas reglas.
- Implementar métricas por campo y análisis de errores.

## Licencias

El código del proyecto podrá publicarse bajo MIT. CORD tiene su propia licencia
CC BY 4.0 y debe descargarse y citarse por separado.


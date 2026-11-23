# Receipt Understanding Benchmark

Banco de pruebas reproducible para extracción de información en tickets. El
proyecto utilizará **CORD** como dataset público y un dataset propio de tickets
españoles, manteniendo un esquema canónico común para todos los modelos.

## Orden experimental

1. VLMs generalistas Qwen3-VL-8B-Instruct y Qwen3-VL-4B-Instruct.
2. Qwen3-VL-4B ajustado para tickets y Donut.
3. Baselines clásicos y con OCR, incluidos reglas sobre OCR ground truth, PaddleOCR + reglas
   y LayoutLMv3.

## Estado de esta versión

- Esquema canónico versionado.
- Mapeo de etiquetas CORD al esquema común.
- Adaptador para anotaciones CORD v1/v2 (`valid_line`).
- Conservación de texto, polígonos, cajas, grupos y orden de lectura.
- CLI para inspeccionar y convertir anotaciones.
- Ejecutor de Qwen 4B/8B y evaluación inicial en CORD v2.
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

## Piloto con Qwen3-VL y CORD v2

En el servidor con una GPU NVIDIA compatible con BF16 y Python 3.10 o posterior:

```bash
python -m pip install -e '.[qwen]'
export HF_HOME="$RECEIPT_DATA_ROOT/models/huggingface"
hf download naver-clova-ix/cord-v2 --repo-type dataset \
  --include 'data/validation-*.parquet' --include 'data/test-*.parquet' \
  --local-dir "$RECEIPT_DATA_ROOT/datasets/cord-v2"

receipt-benchmark run-qwen-cord "$RECEIPT_DATA_ROOT/datasets/cord-v2" \
  "$RECEIPT_DATA_ROOT/predictions/qwen-8b-validation-pilot.jsonl" \
  --model 8b --split validation --limit 15

receipt-benchmark score-cord \
  "$RECEIPT_DATA_ROOT/predictions/qwen-8b-validation-pilot.jsonl"
```

Repite el piloto con `--model 4b` y otro nombre de salida. Para la evaluación final
usa `--split test` sin `--limit` y conserva el mismo prompt y parámetros entre modelos.
La opción `--revision` permite fijar el commit de cada checkpoint.

Cada registro JSONL conserva la salida original, su validez, la predicción y la
referencia proyectadas a los nombres canónicos, la latencia y el pico de memoria
asignada por PyTorch, incluidos los pesos. La latencia incluye el preprocesado y
la generación con lote de uno, tras una pasada de calentamiento. La puntuación
trata los JSON con formato o esquema inválido como predicciones vacías.

La referencia usa `gt_parse` y puntúa las líneas principales de `menu` y el total.
El F1 por campo compara multiconjuntos por ticket y descripción normalizada. El F1
de línea exige coincidencia de los cuatro campos juntos. El acierto del total se
calcula sobre los tickets con total anotado.

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
- Ejecutar el piloto en validación y analizar los errores antes de abrir el test.

## Licencias

El código del proyecto podrá publicarse bajo MIT. CORD tiene su propia licencia
CC BY 4.0 y debe descargarse y citarse por separado.


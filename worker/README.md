# ai-chandra-ocr-worker

GPU-fähiges Dockerimage für Chandra OCR auf RunPod.

Das Image verarbeitet bereits gerenderte PNG/JPEG/WebP/TIFF-Seiten. PDF-Rendering und Upload der PNGs laufen bewusst außerhalb dieses Images, damit keine GPU-Zeit für CPU-Vorverarbeitung verbraucht wird.

## Zielablauf

```text
lokal / CPU:
PDF -> PNG -> S3 Upload

RunPod / GPU:
S3 PNG -> Chandra OCR über vLLM -> S3 Output + State Marker

später:
OCR-Texte -> ChatGPT-Auswertung / inhaltliche Analyse
```

## Vorgeschlagene Ablage im Hauptrepository

```text
docker/ai/chandra-vllm-base/
docker/ai/chandra-ocr-worker/
packaging/docker/ai-chandra-vllm-base.dockerimage.json
packaging/docker/ai-chandra-ocr-worker.dockerimage.json
```

## Image-Aufteilung

```text
ai-chandra-vllm-base:
  CUDA Runtime, Python, vLLM, Chandra OCR und Python-Dependencies

ai-chandra-ocr-worker:
  basiert auf ai-chandra-vllm-base und enthaelt nur Worker-Skript, Entrypoint und README
```

Dadurch muss der schwere vLLM-/Torch-/CUDA-Layer nur selten neu gebaut werden. Aenderungen am Worker erzeugen danach nur noch kleine App-Layer.

## S3-Layout

```text
s3://<bucket>/ocr-jobs/<job-id>/
├─ input/
│  ├─ page-0001.png
│  ├─ page-0002.png
│  └─ page-0003.png
├─ output/
│  └─ page-0001-<hash>/
│     ├─ page-0001.md
│     ├─ page-0001.html
│     ├─ page-0001_metadata.json
│     └─ page-0001_layout.json
└─ state/
   ├─ worker-started.json
   ├─ worker-heartbeat.json
   ├─ page-0001-<hash>.done.json
   ├─ page-0002-<hash>.failed.json
   └─ worker-finished.json
```

Wichtig: Der Done-Marker wird erst nach den Output-Dateien geschrieben. Bei einem Spot-Abbruch kann der nächste Lauf deshalb anhand der Done-Marker sauber fortsetzen.

## Wichtige Environment-Variablen

```text
VLLM_MODEL                         Modell-ID oder lokaler Modellpfad
OCR_MODEL_NAME                     Modellname für OpenAI-kompatible vLLM API, meist gleich VLLM_MODEL
OCR_S3_BUCKET                      Ziel-Bucket
OCR_S3_JOB_PREFIX                  z. B. ocr-jobs/mein-dokument
AWS_REGION / AWS_DEFAULT_REGION    S3 Region
AWS_ACCESS_KEY_ID                  Zugriffsdaten, nicht ins Image einbauen
AWS_SECRET_ACCESS_KEY              Zugriffsdaten, nicht ins Image einbauen
S3_ENDPOINT_URL                    optional für S3-kompatible Endpunkte
```

Defaults:

```text
OCR_S3_INPUT_PREFIX  = <OCR_S3_JOB_PREFIX>/input
OCR_S3_OUTPUT_PREFIX = <OCR_S3_JOB_PREFIX>/output
OCR_S3_STATE_PREFIX  = <OCR_S3_JOB_PREFIX>/state
```

## RunPod und Modellcache

Das Image enthält bewusst keine Modellgewichte. vLLM lädt Modellgewichte beim ersten Start in den Hugging-Face-Cache.

Für RunPod Pods sollte eine persistente Network Volume verwendet werden. Diese ist bei Pods typischerweise unter `/workspace` gemountet. Deshalb zeigen die Cache-Variablen standardmäßig auf `/workspace`:

```text
HF_HOME=/workspace/hf
HF_HUB_CACHE=/workspace/hf/hub
TRANSFORMERS_CACHE=/workspace/hf
XDG_CACHE_HOME=/workspace/cache
VLLM_CACHE_ROOT=/workspace/vllm
TRITON_CACHE_DIR=/workspace/cache/triton
```

Damit muss das Modell nicht bei jedem neuen Pod erneut heruntergeladen werden, solange die gleiche Network Volume wiederverwendet wird.

## Startmodi

Produktiver Worker:

```bash
docker run --gpus all --rm \
  -e VLLM_MODEL="<model>" \
  -e OCR_MODEL_NAME="<model>" \
  -e OCR_S3_BUCKET="<bucket>" \
  -e OCR_S3_JOB_PREFIX="ocr-jobs/example" \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  ai-chandra-ocr-worker:local
```

Debug/Sleep:

```bash
docker run --gpus all --rm -it ai-chandra-ocr-worker:local debug
```

Lokale Verarbeitung ohne S3:

```bash
docker run --gpus all --rm \
  -v "$PWD/in:/data/in" \
  -v "$PWD/out:/data/out" \
  -e VLLM_MODEL="<model>" \
  -e OCR_MODEL_NAME="<model>" \
  ai-chandra-ocr-worker:local worker
```

## Jenkins/Nexus

Die Pipeline sollte zuerst `ai-chandra-vllm-base` bauen und danach `ai-chandra-ocr-worker`. Der Worker hat `dependsOn` auf das Basisimage.

Die Build-Kontexte sind bewusst klein gehalten:

```text
docker/ai/chandra-vllm-base
docker/ai/chandra-ocr-worker
```

Der CI-Smoke-Test sollte klein bleiben:

```bash
python3 -m py_compile /app/run_chandra_on_images.py
bash -n /app/entrypoint.sh
```

Ein echter Modell-/GPU-Test sollte nicht in jedem Jenkins-Build laufen, sondern separat manuell oder als Nightly/RunPod-Test.

## Nicht im Scope dieses Images

- PDF-zu-PNG-Rendering
- Upload der gerenderten PNGs
- ChatGPT-Auswertung der OCR-Texte
- Speicherung von Credentials im Image
- Modellgewichte im Image

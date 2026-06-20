# ai-chandra-vllm-base

Gemeinsames GPU-Basisimage fuer Chandra OCR Worker.

Dieses Image enthaelt CUDA Runtime, Python, vLLM, Chandra OCR und die Python-Abhaengigkeiten. Es enthaelt bewusst keine Modellgewichte und keine Worker-Logik.

Modellgewichte sollen zur Laufzeit in ein persistentes RunPod Network Volume unter `/workspace` geladen werden.

## Enthalten

```text
chandra-ocr
vllm
openai client
boto3 / botocore
pillow
numpy
tqdm
```

## Nicht enthalten

```text
Modellgewichte
S3 Credentials
Worker-Skripte
PDF-zu-PNG Rendering
```

Das eigentliche Worker-Image `ai-chandra-ocr-worker` basiert auf diesem Image und kopiert nur noch die App-Dateien.

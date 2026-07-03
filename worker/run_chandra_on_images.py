#!/usr/bin/env python3
"""
Chandra OCR Runner.

Modes:
- Local mode: process images from --input_dir and write outputs to --output_dir.
- S3 mode: process images from an S3-compatible bucket and write output/state back to S3.

The S3 mode is designed for RunPod spot instances:
- every page is processed independently
- OCR page artifacts are uploaded to output/documents/<ocrDocumentId>/pages/
- input page images are copied to output/documents/<ocrDocumentId>/pages/
- input source PDFs are copied to output/documents/<ocrDocumentId>/source/
- output/_job.json and per-document _job.json files are written for the downstream import-ocr contract
- a done marker is uploaded last
- already completed pages are skipped on restart
- SIGTERM/SIGINT is handled between pages where possible
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from openai import OpenAI
from PIL import Image

from chandra.output import parse_html, parse_layout, parse_markdown
from chandra.prompts import PROMPT_MAPPING


DEFAULT_PROMPT_TYPE = "ocr_layout"
DEFAULT_MAX_SIDE = 1400
DEFAULT_MAX_NEW_TOKENS = 1500
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

_STOP_REQUESTED = False


# -----------------------------
# Generic helpers
# -----------------------------
def request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received signal {signum}. Worker will stop after the current page.", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def join_s3_key(*parts: str) -> str:
    cleaned = [normalize_prefix(p) for p in parts if p is not None and normalize_prefix(str(p))]
    return "/".join(cleaned)


def safe_stem(value: str | Path) -> str:
    stem = Path(str(value)).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "page"


def page_id_from_key(input_key: str, input_prefix: str) -> str:
    rel = input_key
    prefix = normalize_prefix(input_prefix)
    if prefix and input_key.startswith(prefix + "/"):
        rel = input_key[len(prefix) + 1 :]

    stem = safe_stem(Path(rel).name)
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}"


def is_image_key(key: str) -> bool:
    return Path(key).suffix.lower() in IMG_EXTS


@dataclass(frozen=True)
class S3InputPage:
    key: str
    ocr_document_id: str
    page_name: str
    page_num: int


def relative_s3_key(key: str, prefix: str) -> str:
    prefix = normalize_prefix(prefix)
    if prefix and key.startswith(prefix + "/"):
        return key[len(prefix) + 1 :]
    return key


def parse_page_number_from_name(name: str, fallback: int) -> int:
    match = re.search(r"page[-_](\d+)", Path(name).stem, flags=re.IGNORECASE)
    if not match:
        return fallback
    try:
        return max(0, int(match.group(1)) - 1)
    except ValueError:
        return fallback


def s3_input_page_from_key(key: str, input_prefix: str, fallback_index: int) -> Optional[S3InputPage]:
    rel = relative_s3_key(key, input_prefix).replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) < 4:
        return None
    if parts[0] != "documents" or parts[2] != "pages":
        return None
    page_name = parts[-1]
    if not is_image_key(page_name):
        return None
    return S3InputPage(
        key=key,
        ocr_document_id=parts[1],
        page_name=page_name,
        page_num=parse_page_number_from_name(page_name, fallback_index),
    )


def list_s3_v2_input_pages(s3: Any, bucket: str, input_prefix: str) -> List[S3InputPage]:
    documents_prefix = join_s3_key(input_prefix, "documents")
    image_keys = [k for k in s3_list_keys(s3, bucket, documents_prefix) if is_image_key(k)]
    pages: List[S3InputPage] = []
    for idx, key in enumerate(image_keys):
        page = s3_input_page_from_key(key, input_prefix, fallback_index=idx)
        if page is not None:
            pages.append(page)
    return sorted(pages, key=lambda p: (p.ocr_document_id, p.page_num, p.page_name, p.key))


def group_pages_by_document(pages: Sequence[S3InputPage]) -> Dict[str, List[S3InputPage]]:
    grouped: Dict[str, List[S3InputPage]] = {}
    for page in pages:
        grouped.setdefault(page.ocr_document_id, []).append(page)
    return grouped


def list_images(img_dir: str) -> List[Path]:
    p = Path(img_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Input dir does not exist or is not a directory: {p}")
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS]
    return sorted(files)


def resize_if_needed(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img, scale


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sort_layout_blocks_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(b: Dict[str, Any]) -> Tuple[Any, Any]:
        bbox = b.get("bbox") or [0, 0, 0, 0]
        x_min, y_min = bbox[0], bbox[1]
        return (y_min, x_min)

    return sorted(blocks, key=key_fn)


def layout_blocks_to_json(layout_blocks: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not layout_blocks:
        return out
    for block in layout_blocks:
        out.append(
            {
                "bbox": getattr(block, "bbox", None),
                "label": getattr(block, "label", None),
                "content_html": getattr(block, "content", None),
            }
        )
    return out


def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_prompt_text(prompt_type: str, prompt_suffix: str = "") -> str:
    if prompt_type not in PROMPT_MAPPING:
        raise ValueError(f"Unknown prompt_type: {prompt_type}. Allowed: {', '.join(PROMPT_MAPPING.keys())}")
    base = PROMPT_MAPPING[prompt_type].strip()
    if prompt_suffix:
        base = base + "\n\n" + prompt_suffix.strip()
    return base


# -----------------------------
# OCR core
# -----------------------------
@dataclass(frozen=True)
class OcrSettings:
    model_name: str
    prompt_type: str
    max_side: int
    max_new_tokens: int
    include_headers_footers: bool
    include_images_in_output: bool
    write_html_files: bool
    write_metadata_files: bool
    write_layout_json: bool
    save_raw: bool
    vllm_api_base: str
    vllm_api_key: str
    vllm_retries: int
    system_prompt: str
    prompt_suffix: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prompt_type": self.prompt_type,
            "max_side": self.max_side,
            "max_new_tokens": self.max_new_tokens,
            "vllm_api_base": self.vllm_api_base,
            "include_headers_footers": self.include_headers_footers,
            "include_images_in_output": self.include_images_in_output,
            "write_html_files": self.write_html_files,
            "write_metadata_files": self.write_metadata_files,
            "write_layout_json": self.write_layout_json,
            "save_raw": self.save_raw,
            "vllm_retries": self.vllm_retries,
            "system_prompt_enabled": bool(self.system_prompt),
            "prompt_suffix_enabled": bool(self.prompt_suffix),
        }


def make_openai_client(settings: OcrSettings) -> OpenAI:
    api_key = settings.vllm_api_key or os.getenv("VLLM_API_KEY", "EMPTY")
    return OpenAI(api_key=api_key, base_url=settings.vllm_api_base)


def vllm_ocr_call(client: OpenAI, settings: OcrSettings, img: Image.Image) -> Tuple[str, int]:
    prompt_text = build_prompt_text(settings.prompt_type, prompt_suffix=settings.prompt_suffix)
    data_url = image_to_data_url(img, fmt="PNG")

    messages: List[Dict[str, Any]] = []
    if settings.system_prompt:
        messages.append({"role": "system", "content": settings.system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, settings.vllm_retries + 2):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                temperature=0,
                max_tokens=settings.max_new_tokens,
            )
            dt = time.time() - t0
            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            print(f"vLLM seconds: {round(dt, 3)} | completion_tokens: {completion_tokens}", flush=True)
            return content, completion_tokens
        except Exception as exc:
            last_err = exc
            print(f"[vLLM] attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < settings.vllm_retries + 1:
                time.sleep(1.0 * attempt)

    raise RuntimeError(f"vLLM OCR failed after retries: {last_err}")


def process_image(client: OpenAI, img_path: Path, out_dir: Path, settings: OcrSettings, page_num: int) -> Dict[str, Any]:
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    img, scale = resize_if_needed(img, settings.max_side)
    resized_size = img.size

    print(f"[ocr] {img_path.name} orig={orig_size} resized={resized_size} scale={scale:.3f}", flush=True)

    t0 = time.time()
    raw, token_count = vllm_ocr_call(client=client, settings=settings, img=img)
    dt = time.time() - t0

    html = ""
    if settings.write_html_files:
        html = parse_html(
            raw,
            include_images=settings.include_images_in_output,
            include_headers_footers=settings.include_headers_footers,
        )

    md = parse_markdown(
        raw,
        include_images=settings.include_images_in_output,
        include_headers_footers=settings.include_headers_footers,
    )

    layout_blocks_json: List[Dict[str, Any]] = []
    if settings.write_layout_json:
        blocks = parse_layout(raw, img)
        layout_blocks_json = layout_blocks_to_json(blocks)
        layout_blocks_json = sort_layout_blocks_reading_order(layout_blocks_json)

    stem = safe_stem(img_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    write_text(md_path, md)

    html_path: Optional[Path] = None
    if settings.write_html_files:
        html_path = out_dir / f"{stem}.html"
        write_text(html_path, html)

    meta_path: Optional[Path] = None
    if settings.write_metadata_files:
        meta_path = out_dir / f"{stem}_metadata.json"
        page_meta = {
            "file_name": img_path.name,
            "num_pages": 1,
            "total_token_count": token_count,
            "total_chunks": len(layout_blocks_json) if settings.write_layout_json else None,
            "total_images": 0,
            "pages": [
                {
                    "page_num": page_num,
                    "page_box": [0, 0, resized_size[0], resized_size[1]],
                    "token_count": token_count,
                    "num_chunks": len(layout_blocks_json) if settings.write_layout_json else None,
                    "num_images": 0,
                }
            ],
            "run": {
                "created_at_utc": now_utc_iso(),
                "generation_seconds": dt,
                "original_image_size": [orig_size[0], orig_size[1]],
                "resized_image_size": [resized_size[0], resized_size[1]],
                "scale": scale,
                "settings": settings.to_json(),
            },
        }
        write_json(meta_path, page_meta)

    layout_path: Optional[Path] = None
    if settings.write_layout_json:
        layout_path = out_dir / f"{stem}_layout.json"
        layout_payload = {
            "file": img_path.name,
            "page_num": page_num,
            "page_box": [0, 0, resized_size[0], resized_size[1]],
            "num_chunks": len(layout_blocks_json),
            "blocks": layout_blocks_json,
        }
        write_json(layout_path, layout_payload)

    raw_path: Optional[Path] = None
    if settings.save_raw:
        raw_path = out_dir / f"{stem}.raw.txt"
        write_text(raw_path, raw)

    written = [str(p) for p in [md_path, html_path, meta_path, layout_path, raw_path] if p is not None]
    for path in written:
        print(f"[ocr] wrote {path}", flush=True)

    return {
        "input_file": img_path.name,
        "page_num": page_num,
        "token_count": token_count,
        "num_chunks": len(layout_blocks_json) if settings.write_layout_json else None,
        "generation_seconds": dt,
        "original_image_size": [orig_size[0], orig_size[1]],
        "resized_image_size": [resized_size[0], resized_size[1]],
        "scale": scale,
        "outputs": written,
    }


# -----------------------------
# Local mode
# -----------------------------
def run_local(input_dir: str, output_dir: str, settings: OcrSettings, limit: Optional[int]) -> None:
    print("[local] input_dir:", input_dir, flush=True)
    print("[local] output_dir:", output_dir, flush=True)
    print("[local] settings:", json.dumps(settings.to_json(), ensure_ascii=False), flush=True)

    pages = list_images(input_dir)
    if limit is not None:
        pages = pages[: max(0, limit)]
    if not pages:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    client = make_openai_client(settings)
    out_dir = Path(output_dir)
    ocr_document_id = "doc-001-local"
    document_out_dir = out_dir / "documents" / ocr_document_id
    pages_out_dir = document_out_dir / "pages"
    manifest_pages: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(pages):
        if _STOP_REQUESTED:
            print("[local] stop requested before next page.", flush=True)
            break
        print(f"[local] [{idx + 1}/{len(pages)}] {img_path.name}", flush=True)
        result = process_image(client, img_path, pages_out_dir, settings, page_num=idx)
        copied_image = pages_out_dir / img_path.name
        shutil.copy2(img_path, copied_image)
        result["copied_input_image"] = str(copied_image)
        manifest_pages.append(result)

    document_manifest = {
        "version": "ocr-artifacts/v2",
        "artifactVersion": "ocr-artifacts/v2",
        "mode": "local",
        "created_at_utc": now_utc_iso(),
        "ocrDocumentId": ocr_document_id,
        "documentId": ocr_document_id,
        "input_dir": input_dir,
        "output_dir": str(document_out_dir),
        "pages_dir": str(pages_out_dir),
        "source_dir": str(document_out_dir / "source"),
        "settings": settings.to_json(),
        "pages": manifest_pages,
        "stop_requested": _STOP_REQUESTED,
    }

    manifest_payload = {
        "version": "ocr-artifacts/v2",
        "artifactVersion": "ocr-artifacts/v2",
        "mode": "local",
        "created_at_utc": now_utc_iso(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "output_documents_dir": str(out_dir / "documents"),
        "documentCount": 1,
        "documents": [document_manifest],
        "settings": settings.to_json(),
        "stop_requested": _STOP_REQUESTED,
    }

    write_json(document_out_dir / "_job.json", document_manifest)
    write_json(out_dir / "_job.json", manifest_payload)
    write_json(out_dir / "run_manifest.json", manifest_payload)


# -----------------------------
# S3 mode
# -----------------------------
def make_s3_client(endpoint_url: str) -> Any:
    session = boto3.session.Session()
    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "auto"
    return session.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region_name,
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            signature_version="s3v4",
        ),
    )


def s3_key_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def s3_list_keys(s3: Any, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and not key.endswith("/"):
                keys.append(key)
    return sorted(keys)


def s3_upload_file(s3: Any, bucket: str, local_path: Path, key: str, content_type: Optional[str] = None) -> None:
    extra_args: Dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if extra_args:
        s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
    else:
        s3.upload_file(str(local_path), bucket, key)


def s3_put_json(s3: Any, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json; charset=utf-8")


def content_type_for_path(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "text/markdown; charset=utf-8"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".txt":
        return "text/plain; charset=utf-8"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".pdf":
        return "application/pdf"
    return None


def s3_copy_object(s3: Any, bucket: str, source_key: str, target_key: str, content_type: Optional[str] = None) -> None:
    extra_args: Dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type
        extra_args["MetadataDirective"] = "REPLACE"
    s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": source_key}, Key=target_key, **extra_args)


def s3_get_json_optional(s3: Any, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    body = obj["Body"].read().decode("utf-8")
    return json.loads(body)


def copy_input_document_sources(
    s3: Any,
    bucket: str,
    input_prefix: str,
    output_prefix: str,
    ocr_document_id: str,
) -> Tuple[List[str], List[str]]:
    input_source_prefix = join_s3_key(input_prefix, "documents", ocr_document_id, "source")
    output_source_prefix = join_s3_key(output_prefix, "documents", ocr_document_id, "source")
    pdf_keys = [
        k for k in s3_list_keys(s3, bucket, input_source_prefix)
        if Path(k).suffix.lower() == ".pdf"
    ]
    copied_keys: List[str] = []

    for source_key in pdf_keys:
        target_name = "original.pdf" if len(pdf_keys) == 1 else Path(source_key).name
        target_key = join_s3_key(output_source_prefix, target_name)
        s3_copy_object(s3, bucket, source_key, target_key, content_type="application/pdf")
        copied_keys.append(target_key)
        print(f"[s3] copied source PDF s3://{bucket}/{source_key} -> s3://{bucket}/{target_key}", flush=True)

    return pdf_keys, copied_keys


def upload_directory(s3: Any, bucket: str, local_dir: Path, output_prefix: str) -> List[str]:
    uploaded: List[str] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = join_s3_key(output_prefix, rel)
        s3_upload_file(s3, bucket, path, key, content_type_for_path(path))
        uploaded.append(key)
        print(f"[s3] uploaded s3://{bucket}/{key}", flush=True)
    return uploaded


def clean_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            clean_dir(child)
            child.rmdir()
        else:
            child.unlink()


def run_s3(
    bucket: str,
    job_prefix: str,
    input_prefix: str,
    output_prefix: str,
    state_prefix: str,
    download_dir: Path,
    work_dir: Path,
    endpoint_url: str,
    settings: OcrSettings,
    limit: Optional[int],
    force: bool,
    keep_local: bool,
) -> None:
    s3 = make_s3_client(endpoint_url)
    client = make_openai_client(settings)

    input_prefix = normalize_prefix(input_prefix or join_s3_key(job_prefix, "input"))
    output_prefix = normalize_prefix(output_prefix or join_s3_key(job_prefix, "output"))
    state_prefix = normalize_prefix(state_prefix or join_s3_key(job_prefix, "state"))

    print(f"[s3] bucket: {bucket}", flush=True)
    print(f"[s3] input_prefix: {input_prefix}", flush=True)
    print(f"[s3] output_prefix: {output_prefix}", flush=True)
    print(f"[s3] state_prefix: {state_prefix}", flush=True)
    print(f"[s3] endpoint_url: {endpoint_url or '<default>'}", flush=True)

    all_pages = list_s3_v2_input_pages(s3, bucket, input_prefix)
    total_available_input_images = len(all_pages)
    selected_limit = max(0, limit) if limit is not None else None
    pages = all_pages[:selected_limit] if selected_limit is not None else all_pages
    limit_applied = selected_limit is not None and selected_limit < total_available_input_images

    if not pages:
        raise FileNotFoundError(
            f"No v2 input page images found at s3://{bucket}/{join_s3_key(input_prefix, 'documents')}"
        )

    all_document_ids = sorted(group_pages_by_document(all_pages).keys())
    selected_document_ids = sorted(group_pages_by_document(pages).keys())

    download_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    s3_put_json(
        s3,
        bucket,
        join_s3_key(state_prefix, "worker-started.json"),
        {
            "version": "ocr-artifacts/v2",
            "artifactVersion": "ocr-artifacts/v2",
            "created_at_utc": now_utc_iso(),
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "total_available_input_images": total_available_input_images,
            "selected_input_images": len(pages),
            "documentCount": len(all_document_ids),
            "selectedDocumentCount": len(selected_document_ids),
            "limit": limit,
            "limit_applied": limit_applied,
            "settings": settings.to_json(),
        },
    )

    processed = 0
    skipped = 0
    failed = 0
    manifest_pages: List[Dict[str, Any]] = []
    manifest_pages_by_document: Dict[str, List[Dict[str, Any]]] = {doc_id: [] for doc_id in selected_document_ids}

    for idx, page in enumerate(pages, start=1):
        if _STOP_REQUESTED:
            print("[s3] stop requested before next page.", flush=True)
            break

        key = page.key
        ocr_document_id = page.ocr_document_id
        page_id = safe_stem(page.page_name)
        done_key = join_s3_key(state_prefix, "documents", ocr_document_id, f"{page_id}.done.json")
        failed_key = join_s3_key(state_prefix, "documents", ocr_document_id, f"{page_id}.failed.json")
        heartbeat_key = join_s3_key(state_prefix, "worker-heartbeat.json")
        page_output_prefix = join_s3_key(output_prefix, "documents", ocr_document_id, "pages")

        if not force and s3_key_exists(s3, bucket, done_key):
            skipped += 1
            skipped_item = {
                "status": "skipped",
                "input_key": key,
                "ocrDocumentId": ocr_document_id,
                "page_id": page_id,
                "page_num": page.page_num,
                "output_prefix": page_output_prefix,
            }
            manifest_pages.append(skipped_item)
            manifest_pages_by_document.setdefault(ocr_document_id, []).append(skipped_item)
            print(f"[s3] [{idx}/{len(pages)}] skip done: {key}", flush=True)
            continue

        print(f"[s3] [{idx}/{len(pages)}] processing: {key}", flush=True)
        s3_put_json(
            s3,
            bucket,
            heartbeat_key,
            {
                "updated_at_utc": now_utc_iso(),
                "current_input_key": key,
                "current_ocr_document_id": ocr_document_id,
                "current_page_id": page_id,
                "index": idx,
                "total": len(pages),
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
            },
        )

        local_page_dir = work_dir / ocr_document_id / page_id
        local_input = local_page_dir / "in" / page.page_name
        local_output_dir = local_page_dir / "out"
        if not keep_local:
            clean_dir(local_page_dir)
        local_input.parent.mkdir(parents=True, exist_ok=True)
        local_output_dir.mkdir(parents=True, exist_ok=True)

        started_at = now_utc_iso()
        try:
            s3.download_file(bucket, key, str(local_input))
            page_result = process_image(client, local_input, local_output_dir, settings, page_num=page.page_num)
            copied_input_image = local_output_dir / page.page_name
            shutil.copy2(local_input, copied_input_image)
            page_result["copied_input_image"] = str(copied_input_image)
            uploaded_keys = upload_directory(s3, bucket, local_output_dir, page_output_prefix)

            done_payload = {
                "version": "ocr-artifacts/v2",
                "artifactVersion": "ocr-artifacts/v2",
                "status": "done",
                "input_bucket": bucket,
                "input_key": key,
                "ocrDocumentId": ocr_document_id,
                "page_id": page_id,
                "page_num": page.page_num,
                "output_prefix": page_output_prefix,
                "uploaded_keys": uploaded_keys,
                "started_at_utc": started_at,
                "completed_at_utc": now_utc_iso(),
                "result": page_result,
                "settings": settings.to_json(),
            }
            # The done marker is written last. This is the resume boundary.
            s3_put_json(s3, bucket, done_key, done_payload)
            done_item = {
                "status": "done",
                "input_key": key,
                "ocrDocumentId": ocr_document_id,
                "page_id": page_id,
                "page_num": page.page_num,
                "output_prefix": page_output_prefix,
                "uploaded_keys": uploaded_keys,
                "result": page_result,
            }
            manifest_pages.append(done_item)
            manifest_pages_by_document.setdefault(ocr_document_id, []).append(done_item)
            processed += 1
            print(f"[s3] done marker: s3://{bucket}/{done_key}", flush=True)

        except Exception as exc:
            failed += 1
            failure_payload = {
                "version": "ocr-artifacts/v2",
                "artifactVersion": "ocr-artifacts/v2",
                "status": "failed",
                "input_bucket": bucket,
                "input_key": key,
                "ocrDocumentId": ocr_document_id,
                "page_id": page_id,
                "started_at_utc": started_at,
                "failed_at_utc": now_utc_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            s3_put_json(s3, bucket, failed_key, failure_payload)
            print(f"[s3] failed marker: s3://{bucket}/{failed_key}", flush=True)
            raise
        finally:
            if not keep_local:
                try:
                    if local_input.exists():
                        local_input.unlink()
                    clean_dir(local_page_dir)
                except Exception as cleanup_exc:
                    print(f"[s3] cleanup warning: {cleanup_exc}", flush=True)

    source_pdf_input_keys: List[str] = []
    source_pdf_output_keys: List[str] = []
    document_manifests: List[Dict[str, Any]] = []

    for ocr_document_id in selected_document_ids:
        input_document_manifest_key = join_s3_key(input_prefix, "documents", ocr_document_id, "document.json")
        input_document_manifest = s3_get_json_optional(s3, bucket, input_document_manifest_key) or {}
        document_source_input_keys, document_source_output_keys = copy_input_document_sources(
            s3,
            bucket,
            input_prefix,
            output_prefix,
            ocr_document_id,
        )
        source_pdf_input_keys.extend(document_source_input_keys)
        source_pdf_output_keys.extend(document_source_output_keys)

        document_pages = manifest_pages_by_document.get(ocr_document_id, [])
        document_status = "done" if failed == 0 and not _STOP_REQUESTED and not limit_applied else "partial"
        document_manifest = {
            "version": "ocr-artifacts/v2",
            "artifactVersion": "ocr-artifacts/v2",
            "status": document_status,
            "mode": "s3",
            "created_at_utc": now_utc_iso(),
            "bucket": bucket,
            "job_prefix": job_prefix,
            "input_prefix": join_s3_key(input_prefix, "documents", ocr_document_id),
            "output_prefix": join_s3_key(output_prefix, "documents", ocr_document_id),
            "output_pages_prefix": join_s3_key(output_prefix, "documents", ocr_document_id, "pages"),
            "output_source_prefix": join_s3_key(output_prefix, "documents", ocr_document_id, "source"),
            "state_prefix": join_s3_key(state_prefix, "documents", ocr_document_id),
            "ocrDocumentId": ocr_document_id,
            "documentId": ocr_document_id,
            "inputDocumentManifestKey": input_document_manifest_key,
            "inputDocumentManifest": input_document_manifest,
            "originalFilename": input_document_manifest.get("originalFilename"),
            "sourcePdfFileName": input_document_manifest.get("sourcePdfFileName"),
            "sourcePdfSHA256": input_document_manifest.get("sourcePdfSHA256"),
            "source_pdf_input_keys": document_source_input_keys,
            "source_pdf_output_keys": document_source_output_keys,
            "pages": document_pages,
            "settings": settings.to_json(),
            "stop_requested": _STOP_REQUESTED,
            "limit": limit,
            "limit_applied": limit_applied,
        }
        s3_put_json(
            s3,
            bucket,
            join_s3_key(output_prefix, "documents", ocr_document_id, "_job.json"),
            document_manifest,
        )
        document_manifests.append(document_manifest)

    finished_at_utc = now_utc_iso()
    job_status = "done" if failed == 0 and not _STOP_REQUESTED and not limit_applied and processed + skipped == len(pages) else "partial"

    finished_payload = {
        "version": "ocr-artifacts/v2",
        "artifactVersion": "ocr-artifacts/v2",
        "finished_at_utc": finished_at_utc,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "stop_requested": _STOP_REQUESTED,
        "total_available_input_images": total_available_input_images,
        "selected_input_images": len(pages),
        "documentCount": len(all_document_ids),
        "selectedDocumentCount": len(selected_document_ids),
        "limit": limit,
        "limit_applied": limit_applied,
    }

    s3_put_json(
        s3,
        bucket,
        join_s3_key(state_prefix, "worker-finished.json"),
        finished_payload,
    )

    job_manifest = {
        "version": "ocr-artifacts/v2",
        "artifactVersion": "ocr-artifacts/v2",
        "status": job_status,
        "mode": "s3",
        "created_at_utc": finished_at_utc,
        "finished_at_utc": finished_at_utc,
        "bucket": bucket,
        "job_prefix": job_prefix,
        "input_prefix": input_prefix,
        "input_documents_prefix": join_s3_key(input_prefix, "documents"),
        "output_prefix": output_prefix,
        "output_documents_prefix": join_s3_key(output_prefix, "documents"),
        "state_prefix": state_prefix,
        "source_pdf_input_keys": source_pdf_input_keys,
        "source_pdf_output_keys": source_pdf_output_keys,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "stop_requested": _STOP_REQUESTED,
        "total_available_input_images": total_available_input_images,
        "selected_input_images": len(pages),
        "documentCount": len(all_document_ids),
        "selectedDocumentCount": len(selected_document_ids),
        "documentIds": all_document_ids,
        "selectedDocumentIds": selected_document_ids,
        "limit": limit,
        "limit_applied": limit_applied,
        "settings": settings.to_json(),
        "documents": document_manifests,
        "pages": manifest_pages,
    }
    s3_put_json(s3, bucket, join_s3_key(output_prefix, "_job.json"), job_manifest)

    if failed == 0 and not _STOP_REQUESTED and not limit_applied and processed + skipped == len(pages):
        job_done_payload = dict(finished_payload)
        job_done_payload["status"] = "done"
        s3_put_json(
            s3,
            bucket,
            join_s3_key(state_prefix, "job.done.json"),
            job_done_payload,
        )


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chandra OCR locally or as resumable S3 worker.")

    # Local mode
    parser.add_argument("--input_dir", default="", help="Local input image directory.")
    parser.add_argument("--output_dir", default="", help="Local output directory.")

    # S3 mode
    parser.add_argument("--s3_bucket", default="", help="S3 bucket for resumable worker mode.")
    parser.add_argument("--s3_job_prefix", default="", help="Job prefix, for example ocr-jobs/my-document.")
    parser.add_argument("--s3_input_prefix", default="", help="Defaults to <s3_job_prefix>/input.")
    parser.add_argument("--s3_output_prefix", default="", help="Defaults to <s3_job_prefix>/output.")
    parser.add_argument("--s3_state_prefix", default="", help="Defaults to <s3_job_prefix>/state.")
    parser.add_argument("--s3_endpoint_url", default="", help="Optional S3-compatible endpoint URL.")
    parser.add_argument("--s3_download_dir", default="/data/in", help="Local download directory for S3 inputs.")
    parser.add_argument("--s3_work_dir", default="/data/tmp/chandra-ocr-worker", help="Local temporary work directory.")
    parser.add_argument("--s3_keep_local", action="store_true", help="Keep local temp files for debugging.")
    parser.add_argument("--force", action="store_true", help="Reprocess pages even if done markers exist.")

    # OCR settings
    parser.add_argument("--model_name", required=True, help="Model name for vLLM OpenAI API.")
    parser.add_argument("--prompt_type", default=DEFAULT_PROMPT_TYPE, choices=sorted(PROMPT_MAPPING.keys()))
    parser.add_argument("--max_side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    parser.add_argument("--include_headers_footers", action="store_true")
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--no_html", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--layout_json", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--vllm_api_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_api_key", default="")
    parser.add_argument("--vllm_retries", type=int, default=2)
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--prompt_suffix", default="")

    return parser


def settings_from_args(args: argparse.Namespace) -> OcrSettings:
    return OcrSettings(
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        max_side=args.max_side,
        max_new_tokens=args.max_new_tokens,
        include_headers_footers=args.include_headers_footers,
        include_images_in_output=not args.no_images,
        write_html_files=not args.no_html,
        write_metadata_files=not args.no_metadata,
        write_layout_json=args.layout_json,
        save_raw=args.save_raw,
        vllm_api_base=args.vllm_api_base,
        vllm_api_key=args.vllm_api_key,
        vllm_retries=args.vllm_retries,
        system_prompt=args.system_prompt,
        prompt_suffix=args.prompt_suffix,
    )


def main() -> int:
    install_signal_handlers()
    args = build_arg_parser().parse_args()
    settings = settings_from_args(args)
    
    print("Chandra OCR worker starting...")

    if args.s3_bucket:
        if not args.s3_job_prefix and not args.s3_input_prefix:
            raise ValueError("S3 mode requires --s3_job_prefix or --s3_input_prefix.")
        run_s3(
            bucket=args.s3_bucket,
            job_prefix=args.s3_job_prefix,
            input_prefix=args.s3_input_prefix,
            output_prefix=args.s3_output_prefix,
            state_prefix=args.s3_state_prefix,
            download_dir=Path(args.s3_download_dir),
            work_dir=Path(args.s3_work_dir),
            endpoint_url=args.s3_endpoint_url,
            settings=settings,
            limit=args.limit,
            force=args.force,
            keep_local=args.s3_keep_local,
        )
    else:
        if not args.input_dir or not args.output_dir:
            raise ValueError("Local mode requires --input_dir and --output_dir.")
        run_local(args.input_dir, args.output_dir, settings, args.limit)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

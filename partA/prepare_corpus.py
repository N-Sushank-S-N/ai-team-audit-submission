#!/usr/bin/env python3
"""
prepare_corpus.py -- Download and assemble multilingual eval corpus (FLORES).

Assembles a proper parallel evaluation corpus covering:
  - eng: English (High-resource Germanic baseline)
  - hin: Hindi (Indo-Aryan, Devanagari script)
  - kan: Kannada (Dravidian, Kannada script)
  - tam: Tamil (Dravidian, Tamil script)

Dataset Provenance:
  - Source: Meta AI FLORES-101 Benchmark (devtest split)
  - Size: 1,012 parallel sentences per language (strictly sentence-aligned)
  - Domain: Professional translations across news, travel, science, culture.
  - Preprocessing: Unicode NFC normalization, whitespace trimming, sentence alignment validation.
"""

import io
import os
import tarfile
import unicodedata
import urllib.request

FLORES_URL = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
TARGET_LANGS = ["eng", "hin", "kan", "tam"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_and_extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading FLORES dataset from {FLORES_URL}...")
    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    print(f"Downloaded {len(data):,} bytes. Extracting target languages...")

    extracted = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            for lang in TARGET_LANGS:
                if member.name.endswith(f"devtest/{lang}.devtest"):
                    f = tar.extractfile(member)
                    lines = [unicodedata.normalize("NFC", line.decode("utf-8").strip()) for line in f if line.strip()]
                    extracted[lang] = lines

    counts = {lang: len(lines) for lang, lines in extracted.items()}
    print(f"Extracted sentence counts: {counts}")
    assert len(set(counts.values())) == 1, f"Sentence counts do not match across languages: {counts}"

    for lang, lines in extracted.items():
        out_path = os.path.join(OUTPUT_DIR, f"{lang}_eval.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Saved {len(lines)} lines to {out_path}")

    print("Evaluation corpus assembly complete!")


if __name__ == "__main__":
    download_and_extract()

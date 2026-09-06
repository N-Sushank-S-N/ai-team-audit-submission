#!/usr/bin/env python3
r"""
fertility_fixed.py -- Corrected, production-grade tokenizer benchmark.

Computes tokenizer fertility and compression metrics across multiple languages
using mathematically sound micro-averaging and multi-denominator analysis:
  - Tokens per whitespace word (proper split)
  - Tokens per grapheme cluster (aksharas / visual characters via \X)
  - Tokens per UTF-8 byte
  - Tokens per parallel sentence (semantic unit holding meaning constant)
  - Tokens per Unicode codepoint

Supports:
  - tiktoken encodings (e.g. gpt2, cl100k_base, o200k_base)
  - HuggingFace tokenizers (e.g. hf:xlm-roberta-base, hf:Qwen/Qwen2.5-0.5B)
r"""

import argparse
import csv
import json
import os
import sys
import unicodedata
import regex

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken

        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def analyze(lines, encode):
    r"""
    Computes rigorous tokenizer metrics over a list of sentences.
    Returns micro-averaged (corpus-level) and macro-averaged metrics.
    r"""
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0
    total_codepoints = 0
    num_sentences = len(lines)

    per_line_tok_word = []
    per_line_tok_grapheme = []
    per_line_tok_byte = []
    per_line_tok_sent = []

    for line in lines:
        tokens = encode(line)
        num_tok = len(tokens)
        words = line.split()  # Correct split handling consecutive spaces
        num_words = len(words)
        graphemes = regex.findall(r"\X", line)
        num_graphemes = len(graphemes)
        num_bytes = len(line.encode("utf-8"))
        num_chars = len(line)

        total_tokens += num_tok
        total_words += num_words
        total_graphemes += num_graphemes
        total_bytes += num_bytes
        total_codepoints += num_chars

        if num_words > 0:
            per_line_tok_word.append(num_tok / num_words)
        if num_graphemes > 0:
            per_line_tok_grapheme.append(num_tok / num_graphemes)
        if num_bytes > 0:
            per_line_tok_byte.append(num_tok / num_bytes)
        per_line_tok_sent.append(num_tok)

    # Micro-averages (Ratio of sums: true corpus level)
    micro = {
        "tok_per_sentence": total_tokens / num_sentences if num_sentences else 0,
        "tok_per_word": total_tokens / total_words if total_words else 0,
        "tok_per_grapheme": total_tokens / total_graphemes if total_graphemes else 0,
        "tok_per_byte": total_tokens / total_bytes if total_bytes else 0,
        "tok_per_char": total_tokens / total_codepoints if total_codepoints else 0,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_graphemes": total_graphemes,
        "total_bytes": total_bytes,
        "total_chars": total_codepoints,
        "sentences": num_sentences,
    }

    # Macro-averages (Average of ratios)
    macro = {
        "tok_per_word": sum(per_line_tok_word) / len(per_line_tok_word) if per_line_tok_word else 0,
        "tok_per_grapheme": sum(per_line_tok_grapheme) / len(per_line_tok_grapheme) if per_line_tok_grapheme else 0,
        "tok_per_byte": sum(per_line_tok_byte) / len(per_line_tok_byte) if per_line_tok_byte else 0,
    }

    return micro, macro


def main():
    ap = argparse.ArgumentParser(description="Corrected tokenizer benchmark across multiple denominators.")
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and file path, e.g. eng=data/eng_eval.txt (repeatable)",
    )
    ap.add_argument("--tokenizer", default="gpt2", help="tiktoken encoding or hf:<repo_id>")
    ap.add_argument("--output-json", default=None, help="Path to save JSON results")
    ap.add_argument("--output-csv", default=None, help="Path to save CSV results")
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    results = {}
    print(f"\nTokenizer: {args.tokenizer}")
    print(f"{'lang':<8}{'tok/sent':>12}{'tok/word':>12}{'tok/graph':>12}{'tok/byte':>12}{'tok/char':>12}")
    print("-" * 68)

    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        micro, macro = analyze(lines, encode)
        results[lang] = {"micro": micro, "macro": macro}
        print(
            f"{lang:<8}"
            f"{micro['tok_per_sentence']:>12.2f}"
            f"{micro['tok_per_word']:>12.2f}"
            f"{micro['tok_per_grapheme']:>12.2f}"
            f"{micro['tok_per_byte']:>12.3f}"
            f"{micro['tok_per_char']:>12.3f}"
        )

    # Relative cross-language ratios against base language
    if len(results) >= 2:
        langs = list(results)
        base = langs[0]
        print(f"\nRelative Cross-Language Expansion Ratios (relative to '{base}'):")
        print(f"{'lang':<8}{'per sent':>14}{'per word':>14}{'per graph':>14}{'per byte':>14}")
        print("-" * 64)
        for lang in langs:
            base_m = results[base]["micro"]
            m = results[lang]["micro"]
            r_sent = m["tok_per_sentence"] / base_m["tok_per_sentence"]
            r_word = m["tok_per_word"] / base_m["tok_per_word"]
            r_graph = m["tok_per_grapheme"] / base_m["tok_per_grapheme"]
            r_byte = m["tok_per_byte"] / base_m["tok_per_byte"]
            print(f"{lang:<8}{r_sent:>14.2f}x{r_word:>14.2f}x{r_graph:>14.2f}x{r_byte:>14.2f}x")

    # Export outputs if requested
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": args.tokenizer, "results": results}, f, indent=2)
        print(f"\nResults exported to JSON: {args.output_json}")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tokenizer", "lang", "tok_per_sentence", "tok_per_word", "tok_per_grapheme", "tok_per_byte", "tok_per_char", "total_tokens", "total_words", "total_bytes"])
            for lang, data in results.items():
                m = data["micro"]
                writer.writerow([
                    args.tokenizer,
                    lang,
                    f"{m['tok_per_sentence']:.2f}",
                    f"{m['tok_per_word']:.2f}",
                    f"{m['tok_per_grapheme']:.2f}",
                    f"{m['tok_per_byte']:.4f}",
                    f"{m['tok_per_char']:.4f}",
                    m["total_tokens"],
                    m["total_words"],
                    m["total_bytes"],
                ])
        print(f"Results exported to CSV: {args.output_csv}")


if __name__ == "__main__":
    main()


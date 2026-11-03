#!/usr/bin/env python3
"""Fetch the third-party corpora and arrival traces this work reads.

None of these datasets is redistributed in this repository. Each is obtained from its own
published release, under its own licence, into data/. See data/README.md for provenance.

    python scripts/fetch_data.py            # fetch everything that is missing
    python scripts/fetch_data.py otb        # fetch one source by name
    python scripts/fetch_data.py --list     # show what is expected and what is present

Nothing here is needed to reproduce the results. The model generations, judge decisions and
oracle labels derived from these corpora are committed under results/paired/ and
results/crossdataset/, so the routers, the serving simulations and every figure can be rebuilt
without obtaining the corpora at all. Fetch them only to repeat the generation step itself, or
to re-run the two experiments that read the production arrival traces.
"""
import argparse
import os
import pathlib
import sys
import urllib.error
import urllib.request

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# Azure publishes the LLM inference traces in its public dataset repository. Only the arrival
# timestamps are read; see the note in data/README.md.
AZURE_REPO = "https://github.com/Azure/AzurePublicDataset"
AZURE_RAW = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"

MOHSENI_URL = ("https://huggingface.co/datasets/AmirMohseni/reasoning-router-data-v2"
               "/resolve/main/data/test-00000-of-00001.parquet")

SOURCES = {
    "otb": {
        "target": DATA / "otb" / "train.parquet",
        "what": "OptimalThinkingBench (facebook/optimal_thinking_bench), train split",
        "how": "hf",
        "hf": ("facebook/optimal_thinking_bench", None, "train"),
    },
    "mohseni": {
        "target": DATA / "mohseni" / "test.parquet",
        "what": "Mohseni reasoning-router-data-v2, test split",
        "how": "url",
        "url": MOHSENI_URL,
    },
    "gsm8k": {
        "target": DATA / "gsm8k" / "test.parquet",
        "what": "GSM8K (main config), test split",
        "how": "hf",
        "hf": ("gsm8k", "main", "test"),
    },
    "azure-conv": {
        "target": DATA / "azure" / "AzureLLMInferenceTrace_conv.csv",
        "what": "Azure LLM inference trace, conversation service",
        "how": "url",
        "url": f"{AZURE_RAW}/AzureLLMInferenceTrace_conv.csv",
        "fallback": AZURE_REPO,
    },
    "azure-code": {
        "target": DATA / "azure" / "AzureLLMInferenceTrace_code.csv",
        "what": "Azure LLM inference trace, code-completion service",
        "how": "url",
        "url": f"{AZURE_RAW}/AzureLLMInferenceTrace_code.csv",
        "fallback": AZURE_REPO,
    },
}


def fetch_url(url, target, fallback=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
            while chunk := r.read(1 << 16):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        msg = f"HTTP {e.code} for {url}"
        if fallback:
            msg += (f"\n      The published layout may have changed. Download the file by hand"
                    f"\n      from {fallback} and place it at {target}")
        raise RuntimeError(msg) from None
    tmp.replace(target)
    return target.stat().st_size


def fetch_hf(repo, config, split, target):
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "needs the 'datasets' package:  pip install datasets\n"
            f"      or download {repo} ({split} split) by hand to {target}") from None
    ds = load_dataset(repo, config)[split] if config else load_dataset(repo)[split]
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(target))
    return target.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of: {', '.join(SOURCES)}")
    ap.add_argument("--list", action="store_true", help="show expected files and exit")
    ap.add_argument("--force", action="store_true", help="re-fetch files already present")
    args = ap.parse_args()

    if args.list:
        for name, s in SOURCES.items():
            rel = s["target"].relative_to(DATA.parent)
            mark = "present" if s["target"].exists() else "missing"
            print(f"  {name:12} [{mark:7}] {rel}\n               {s['what']}")
        return 0

    unknown = [n for n in args.names if n not in SOURCES]
    if unknown:
        print(f"unknown source(s): {', '.join(unknown)}\nknown: {', '.join(SOURCES)}",
              file=sys.stderr)
        return 2

    todo = args.names or list(SOURCES)
    failed = []
    for name in todo:
        s = SOURCES[name]
        target = s["target"]
        if target.exists() and not args.force:
            print(f"  {name:12} already present, skipping ({target.name})")
            continue
        print(f"  {name:12} fetching {s['what']} ...")
        try:
            if s["how"] == "url":
                size = fetch_url(s["url"], target, s.get("fallback"))
            else:
                size = fetch_hf(*s["hf"], target)
        except Exception as e:
            print(f"  {'':12} FAILED: {e}", file=sys.stderr)
            failed.append(name)
            continue
        print(f"  {'':12} wrote {target.relative_to(DATA.parent)} ({size / 1e6:.1f} MB)")

    if failed:
        print(f"\n{len(failed)} of {len(todo)} failed: {', '.join(failed)}", file=sys.stderr)
        print("The committed results under results/ do not depend on these files.",
              file=sys.stderr)
        return 1
    print("\nAll requested sources present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

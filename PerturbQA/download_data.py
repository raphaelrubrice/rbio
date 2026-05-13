#!/usr/bin/env python3
"""Download PerturbQA KG + gene summaries from Zenodo.

Usage:
    python PerturbQA/download_data.py              # writes into PerturbQA/ by default
    python PerturbQA/download_data.py --dest /content/PerturbQA   # Colab
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://zenodo.org/records/14915313/files"
ASSETS = {
    "kg.zip":           "kg",
    "gene_summary.zip": "gene_summary",
    "README.md":        None,
}
REQUIRED_KG = [
    "bioplex.json", "corum.json", "corum_gsea.json", "ensembl.json",
    "go.json", "go_dict.json", "go_gsea.json", "reactome.json",
    "reactome_gsea.json", "string.json", "uniprot.json",
]
REQUIRED_SUMMARIES = ["desc_gene.json", "desc_pert.json"]


def download(url, label):
    print(f"  Downloading {label} ...", end=" ", flush=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    print(f"{len(data) // 1024 // 1024} MB")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Download PerturbQA data from Zenodo."
    )
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).parent),
        help="Destination directory (default: PerturbQA/ folder next to this script)",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for filename, subdir in ASSETS.items():
        url = f"{BASE_URL}/{filename}?download=1"
        if subdir is not None and (dest / subdir).exists():
            print(f"  {subdir}/ already exists — skipping.")
            continue
        data = download(url, filename)
        if subdir is None:
            (dest / filename).write_bytes(data)
        else:
            print(f"  Extracting → {dest / subdir}/")
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest)

    missing_kg  = [f for f in REQUIRED_KG       if not (dest / "kg" / f).exists()]
    missing_sum = [f for f in REQUIRED_SUMMARIES if not (dest / "gene_summary" / f).exists()]

    if missing_kg or missing_sum:
        print("\nWARNING — missing files after download:")
        for f in missing_kg:  print(f"  kg/{f}")
        for f in missing_sum: print(f"  gene_summary/{f}")
        sys.exit(1)

    print("\nPerturbQA data ready.")
    print(f"  KG files       → {dest / 'kg'}")
    print(f"  Gene summaries → {dest / 'gene_summary'}")


if __name__ == "__main__":
    main()

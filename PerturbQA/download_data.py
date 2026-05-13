#!/usr/bin/env python3
"""Download PerturbQA KG + gene summaries from Google Drive.

Usage:
    python PerturbQA/download_data.py              # writes into PerturbQA/ by default
    python PerturbQA/download_data.py --dest /content/rbio/PerturbQA   # Colab
"""
import argparse
import sys
import zipfile
from pathlib import Path

import gdown

ASSETS = {
    "kg.zip": {
        "url":    "https://drive.google.com/uc?id=1czfVavbkkZFBNLYMCRIICh2Czr-LOfSU",
        "subdir": "kg",
    },
    "gene_summary.zip": {
        "url":    "https://drive.google.com/uc?id=1sdh14LnF3unPWZiTmYYOMN29FH3McIUX",
        "subdir": "gene_summary",
    },
}
REQUIRED_KG = [
    "bioplex.json", "corum.json", "corum_gsea.json", "ensembl.json",
    "go.json", "go_dict.json", "go_gsea.json", "reactome.json",
    "reactome_gsea.json", "string.json", "uniprot.json",
]
REQUIRED_SUMMARIES = ["desc_gene.json", "desc_pert.json"]


def main():
    parser = argparse.ArgumentParser(
        description="Download PerturbQA data from Google Drive."
    )
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).parent),
        help="Destination directory (default: PerturbQA/ folder next to this script)",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for filename, spec in ASSETS.items():
        subdir = spec["subdir"]
        if (dest / subdir).exists():
            print(f"  {subdir}/ already exists — skipping.")
            continue
        tmp = dest / filename
        print(f"  Downloading {filename} ...")
        gdown.download(spec["url"], str(tmp), quiet=False)
        print(f"  Extracting → {dest / subdir}/")
        with zipfile.ZipFile(tmp) as z:
            z.extractall(dest)
        tmp.unlink()

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

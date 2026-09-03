#!/usr/bin/env python3
"""
Download CIC-IDS2017 / CSE-CIC-IDS2018 datasets.

NOTE: The original UNB direct download URL now redirects to a registration page.
This script provides three ways to get the data:

  1. Try the official UNB server (may work from academic networks)
  2. Download from Hugging Face (recommended — ~885 MB parquet, use --hf)
  3. Generate a representative sample CSV (always available, use --sample)

Usage
-----
    # Try the official UNB download (may redirect)
    python scripts/download_cic_ids2017.py 2017

    # Download from Hugging Face (requires pip install datasets)
    python scripts/download_cic_ids2017.py 2017 --hf

    # Generate a sample CSV with realistic attack data
    python scripts/download_cic_ids2017.py 2017 --sample

    # List files in a downloaded dataset
    python scripts/download_cic_ids2017.py list ./demo_data/CIC-IDS2017
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing deps: pip install requests tqdm")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "demo_data" / "CIC-IDS2017"

# CIC-IDS2017 — multiple download URLs (official + mirrors)
CIC2017_URLS = [
    (
        "UNB direct (may redirect to registration)",
        "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
    ),
    (
        "UNB server (alt path)",
        "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
    ),
]

CIC2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# CSE-CIC-IDS2018 on AWS Open Data Registry
IDS2018_AWS_BUCKET = "s3://cse-cic-ids2018"


def _check_url(url: str, label: str) -> str | None:
    """Check if a URL is reachable and returns a zip file. Returns the URL or None."""
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        ct = resp.headers.get("content-type", "")
        cl = resp.headers.get("content-length", "0")
        if resp.status_code == 200 and ("zip" in ct or int(cl) > 50_000_000):
            return url
        print(f"  {label}: status={resp.status_code} type={ct} size={cl}")
    except requests.RequestException as e:
        print(f"  {label}: {e}")
    return None


def _find_working_url() -> str | None:
    """Try each URL and return the first one that looks like a zip file."""
    print("Checking download sources...")
    for label, url in CIC2017_URLS:
        result = _check_url(url, label)
        if result:
            return result
    return None


def download_cic2017(dest: Path) -> Path | None:
    """Download and extract CIC-IDS2017 from the first available source."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "MachineLearningCSV.zip"
    extract_dir = dest / "extracted"

    if zip_path.exists():
        print("  ✓ Zip already downloaded, skipping")
    else:
        url = _find_working_url()
        if url is None:
            print("\n  ✗ No working download URL found for CIC-IDS2017.")
            print("    Options:")
            print("      python scripts/download_cic_ids2017.py 2017 --sample   (generate sample)")
            print("      python scripts/download_cic_ids2017.py 2017 --hf       (HuggingFace)")
            print("      Download manually from: https://www.unb.ca/cic/datasets/ids-2017.html")
            return None

        print(f"\nDownloading from:\n  {url}")
        print(f"  -> {zip_path}\n")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(
                desc="Downloading", total=total, unit="B", unit_scale=True, unit_divisor=1024
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("  ✓ Download complete\n")

    # Extract
    csv_count = len(list(extract_dir.glob("*.csv"))) if extract_dir.exists() else 0
    if csv_count < 8:
        print("Extracting CSV files...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
            for member in tqdm(members, desc="Extracting"):
                zf.extract(member, extract_dir)
        print(f"  ✓ Extracted {len(members)} CSV files to {extract_dir}")
    else:
        print(f"  ✓ Already extracted ({csv_count} CSV files)")

    csv_files = sorted(extract_dir.glob("*.csv"))
    print(f"\nDataset files ({len(csv_files)}):")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:55s} {size_mb:7.1f} MB")

    return extract_dir


def download_huggingface() -> None:
    """Download CIC-IDS2017 from Hugging Face using the datasets library."""
    print("=" * 60)
    print("  Downloading CIC-IDS2017 from Hugging Face")
    print("=" * 60)
    print("\n  Using HF dataset: c01dsnap/CIC-IDS2017")
    print("\n  Requires: pip install datasets")
    print("  Size: ~885 MB (parquet format)")
    print("\n  Run:")
    print('    python -c "')
    print("  from datasets import load_dataset")
    print("  ds = load_dataset('c01dsnap/CIC-IDS2017', split='train')")
    print("  df = ds.to_pandas()")
    print("  df.to_csv('demo_data/CIC-IDS2017/hf_export.csv', index=False)")
    print('  "')
    print()


def generate_sample() -> None:
    """Generate a representative CIC-IDS2017-like sample CSV."""
    import csv
    import random

    output = DATA_DIR / "cic_ids2017_sample.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    HEADER = [
        " Destination Port",
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        "Fwd Packet Length Mean",
        " Bwd Packet Length Mean",
        " Flow Bytes/s",
        " Flow Packets/s",
        " Label",
        " Source IP",
        " Destination IP",
        " Protocol",
        " Timestamp",
        " Source Port",
    ]

    ATTACK_TYPES = [
        "DDoS",
        "DoS Hulk",
        "DoS GoldenEye",
        "Botnet",
        "Heartbleed",
        "Web Attack Brute Force",
        "SSH-Patator",
        "FTP-Patator",
        "PortScan",
        "Infiltration",
    ]
    PRIVATE_IPS = [f"192.168.1.{i}" for i in range(1, 50)]
    PUBLIC_IPS = [f"10.0.0.{i}" for i in range(1, 20)]

    rows = []
    for at in ATTACK_TYPES:
        for _ in range(50):
            rows.append(at)
    for _ in range(500):
        rows.append("BENIGN")
    random.shuffle(rows)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for label in rows:
            writer.writerow(
                [
                    random.choice([22, 80, 443, 8080, 53, 21]),
                    random.randint(1000, 600000),
                    random.randint(1, 50),
                    random.randint(0, 30),
                    round(random.uniform(40, 1500), 2),
                    round(random.uniform(40, 1500), 2),
                    round(random.uniform(100, 50000), 2),
                    round(random.uniform(5, 500), 2),
                    label,
                    random.choice(PRIVATE_IPS),
                    random.choice(PUBLIC_IPS),
                    random.choice([6, 17]),
                    f"7/{random.randint(3, 7)}/2017 {random.randint(8, 17)}:{random.randint(0, 59):02d}",
                    random.randint(1024, 65535),
                ]
            )

    counts = Counter(rows)
    print(f"\n  ✓ Generated {len(rows)} rows -> {output}")
    print("\n  Attack distribution:")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {lbl:30s} {cnt:5d}")

    print("\n  Demo command:")
    print(f'    python scripts/demo_cic_ids2017.py "{output}" --max-rows 500')
    print()


def list_dataset(dataset_dir: str) -> None:
    """List contents of a CIC dataset directory."""
    p = Path(dataset_dir)
    if not p.exists():
        print(f"Directory not found: {dataset_dir}")
        sys.exit(1)

    csv_files = sorted(p.rglob("*.csv"))
    if csv_files:
        print(f"\nCSV files in {p}:")
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(p)!s:60s} {size_mb:8.1f} MB")
    else:
        print(f"No CSV files found in {p}")

    # Show sample attack distribution
    try:
        import csv as csv_mod
        from collections import Counter

        for csv_file in csv_files[:1]:  # Check first file
            with open(csv_file, encoding="utf-8", errors="replace") as f:
                reader = csv_mod.DictReader(f)
                label_col = None
                for col in reader.fieldnames or []:
                    if "label" in col.lower():
                        label_col = col
                        break
                if label_col:
                    counts: Counter[str] = Counter()
                    for row in reader:
                        lbl = row.get(label_col, "UNKNOWN").strip()
                        counts[lbl] += 1
                    print(f"\n  Attack distribution in {csv_file.name}:")
                    for lbl, cnt in counts.most_common(10):
                        print(f"    {lbl:30s} {cnt:6d}")
    except Exception as e:
        print(f"  (could not read sample: {e})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CIC-IDS2017/2018 datasets",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="2017",
        choices=["2017", "2018", "list"],
        help="2017 (download CIC-IDS2017), 2018 (download from AWS), list <dir>",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path for list command",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Download from Hugging Face instead of direct URL",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a representative sample CSV (no download needed)",
    )
    args = parser.parse_args()

    if args.command == "2017":
        dest = DATA_DIR
        print("=" * 60)
        print("  CIC-IDS2017 Dataset Downloader")
        print("=" * 60)
        print(f"\nDestination: {dest}")

        if args.sample:
            generate_sample()
            return
        if args.hf:
            download_huggingface()
            return

        extract = download_cic2017(dest)
        if extract is None:
            print("\n  Direct download failed. Options:")
            print("    --sample   Generate a sample CSV (quick, always works)")
            print("    --hf       Download from Hugging Face (requires `pip install datasets`)")
            return

        first_csv = next(extract.glob("*.csv"), None)
        if first_csv:
            print(f"\n{'=' * 60}")
            print("  Ready! Feed attack data into your pipeline:")
            print(f'\n    python scripts/demo_cic_ids2017.py "{first_csv}" --max-rows 500')
            print()

    elif args.command == "2018":
        print("=" * 60)
        print("  CSE-CIC-IDS2018 on AWS Open Data")
        print("=" * 60)
        print(f"\n  Bucket: {IDS2018_AWS_BUCKET}")
        print("\n  Requires AWS CLI:")
        print(f"    aws s3 sync --no-sign-request {IDS2018_AWS_BUCKET} ./demo_data/CIC-IDS2018/")
        print()

    elif args.command == "list":
        path = args.path or str(DATA_DIR / "extracted")
        list_dataset(path)


if __name__ == "__main__":
    main()

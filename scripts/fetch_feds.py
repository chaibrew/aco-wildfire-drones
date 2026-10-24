"""Download the published FEDS archive, verify it, extract one year's GeoPackage."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/19170071/versions/1"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=range(2012, 2021), default=2018)
    args = parser.parse_args()
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=60) as response:
        metadata = json.load(response)
    entry = next(f for f in metadata["files"] if f["name"] == "Largefire.tar.gz")
    archive = raw / entry["name"]
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print(f"Downloading {entry['size'] / 1e6:.1f} MB from Figshare", flush=True)
        with urllib.request.urlopen(entry["download_url"], timeout=120) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != entry["computed_md5"]:
        raise ValueError(f"Checksum mismatch for {archive}; remove it and retry")
    name = f"LargeFires_{args.year}.gpkg"
    # Copy only this known regular file, never extract arbitrary archive paths.
    with tarfile.open(archive) as package:
        member = package.getmember(f"Largefire/{name}")
        if not member.isfile():
            raise ValueError("Expected a regular GeoPackage file")
        with package.extractfile(member) as src, (raw / name).open("wb") as dst:
            shutil.copyfileobj(src, dst)
    with (raw / name).open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"metadata_url": ARTICLE, "download_url": entry["download_url"],
                "archive_md5": md5, "extracted_file": name, "extracted_sha256": sha256,
                "license": metadata["license"], "citation": "https://doi.org/10.1038/s41597-022-01343-0"}
    (raw / f"{name}.provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(raw / name)


if __name__ == "__main__":
    main()

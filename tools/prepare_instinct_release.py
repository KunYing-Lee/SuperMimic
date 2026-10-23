"""Fetch immutable InstinctMJ source robot and released motion/scene pairs.

The release uses a torso-root G1. A pelvis-root G1 gives incorrect world
coordinates even when all 29 joint names match. No pose or mesh is corrected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

REVISION = "c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b"
RELEASE = (
    "https://github.com/project-instinct/InstinctMJ/releases/download/supermimic-v0.1/"
)
ARCHIVE = "20251116_50cm_kneeClimbStep1.zip"
ROBOT = "g1_29dof_torsobase_popsicle.xml"
RAW = f"https://raw.githubusercontent.com/project-instinct/InstinctMJ/{REVISION}/src/instinct_mj/assets/resources/unitree_g1/"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        path.write_bytes(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/instinct_release"))
    args = p.parse_args()
    root = args.output.resolve()
    download(RELEASE + ARCHIVE, root / ARCHIVE)
    download(RELEASE + "SHA256SUMS", root / "SHA256SUMS")
    checksums = dict(
        (line.split()[-1].lstrip("*"), line.split()[0])
        for line in (root / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    )
    if checksums[ARCHIVE] != sha(root / ARCHIVE):
        raise ValueError("Published release checksum mismatch")
    extracted = root / "unpacked"
    with zipfile.ZipFile(root / ARCHIVE) as archive:
        for name in archive.namelist():
            if not (extracted / name).resolve().is_relative_to(extracted.resolve()):
                raise ValueError(f"Unsafe archive path: {name}")
        archive.extractall(extracted)
    xml = root / "g1" / "xml" / ROBOT
    download(RAW + "xml/" + ROBOT, xml)
    meshes = [x.attrib["file"] for x in ET.parse(xml).findall("./asset/mesh")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(download, RAW + "meshes/" + name, root / "g1" / "meshes" / name)
            for name in meshes
        ]
        for f in futures:
            f.result()
    paths = [
        root / ARCHIVE,
        root / "SHA256SUMS",
        xml,
        *sorted((root / "g1" / "meshes").iterdir()),
    ]
    provenance = {
        "release": RELEASE,
        "source_revision": REVISION,
        "source_root_body": "torso_link",
        "transformations": [],
        "sha256": {str(f.relative_to(root)): sha(f) for f in paths},
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(root),
                "release_sha256": sha(root / ARCHIVE),
                "robot_assets": len(meshes),
            }
        )
    )


if __name__ == "__main__":
    main()

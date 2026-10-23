"""Materialize the pinned K1Lab serial-ankle MJCF and record every asset hash.

The source robot model is preserved byte-for-byte. This does not assert that
an external retargeted dataset used the same kinematics: import validation
must independently compare its supplied body poses with this model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

SOURCE_REPOSITORY = "https://github.com/KunYing-Lee/k1lab"
SOURCE_REVISION = "30f015a61f9583cba664b5ea65b0a1c73fba57e2"
SOURCE_DIRECTORY = "src/k1lab/robots/k1/xmls"
ROOT = Path(__file__).resolve().parents[1]


def prepare(checkout: Path, output: Path):
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing asset directory: {output}"
        )

    def read_git(path: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(checkout), "show", f"{SOURCE_REVISION}:{path}"],
            check=True,
            capture_output=True,
        ).stdout

    xml = read_git(f"{SOURCE_DIRECTORY}/k1.xml")
    tree = ET.fromstring(xml)
    files = {"k1.xml": xml}
    for mesh in tree.findall("./asset/mesh"):
        filename = Path(mesh.attrib["file"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError(f"Unexpected mesh path: {filename}")
        relative = f"meshes/{filename.as_posix()}"
        files[relative] = read_git(f"{SOURCE_DIRECTORY}/{relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="k1-assets-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "robot"
        staged.mkdir()
        records = []
        for name, data in files.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        model = mujoco.MjModel.from_xml_path(str(staged / "k1.xml"))
        if model.nq != 29 or model.nv != 28:
            raise ValueError(
                f"Expected K1 22 joints plus floating base, got nq={model.nq}, nv={model.nv}"
            )
        manifest = {
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_directory": SOURCE_DIRECTORY,
            "files": records,
            "model": {
                "nq": model.nq,
                "nv": model.nv,
                "nbody": model.nbody,
                "mass_kg": float(model.body_mass.sum()),
                "joint_order": [
                    model.joint(j).name
                    for j in range(model.njnt)
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
                ],
            },
            "transformations": [],
        }
        (staged / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.move(staged, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        required=True,
        help=f"Local {SOURCE_REPOSITORY} checkout containing {SOURCE_REVISION}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "k1_description"
    )
    args = parser.parse_args()
    manifest = prepare(args.source_checkout.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "source_revision": SOURCE_REVISION,
                "files": len(manifest["files"]),
                "model": manifest["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

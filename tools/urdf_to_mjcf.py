"""Compile the retargeting URDF to floating-base MJCF without changing inertias."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def convert(source: Path, output: Path) -> mujoco.MjModel:
    tree = ET.parse(source)
    root = tree.getroot()
    extension = root.find("mujoco")
    if extension is not None:
        root.remove(extension)
    extension = ET.SubElement(root, "mujoco")
    # Keep named link frames and the original inertias. Invalid inertias fail
    # compilation rather than being silently replaced by balanceinertia.
    ET.SubElement(extension, "compiler", meshdir=str(source.parent),
                  discardvisual="false", fusestatic="false", strippath="false",
                  balanceinertia="false")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supermimic-urdf-") as tmp:
        urdf = Path(tmp) / source.name
        tree.write(urdf, encoding="utf-8", xml_declaration=True)
        original = mujoco.MjModel.from_xml_path(str(urdf))
        compiled = Path(tmp) / "compiled.xml"
        mujoco.mj_saveLastXML(str(compiled), original)
        mjcf = ET.parse(compiled)
        pelvis = mjcf.getroot().find("./worldbody/body[@name='pelvis']")
        if pelvis is None:
            raise ValueError("The source URDF must have the named pelvis root body")
        pelvis.insert(0, ET.Element("freejoint", name="floating_base_joint"))
        # Keep the generated model relocatable with its source mesh directory.
        compiler = mjcf.getroot().find("compiler")
        compiler.set("meshdir", os.path.relpath(source.parent, output.parent))
        mjcf.write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Expected 29 joints + floating base, got nq={model.nq}, nv={model.nv}")
    np.testing.assert_allclose(model.body_mass, original.body_mass, rtol=0, atol=1e-6)
    np.testing.assert_allclose(model.body_inertia, original.body_inertia, rtol=1e-5, atol=1e-9)
    print(f"MJCF: nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom} mass={model.body_mass.sum():.9g}")
    return model


def main():
    default_dir = Path(__file__).resolve().parents[1] / "data" / "g1_description"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_dir / "g1_29dof_rev_1_0.urdf")
    parser.add_argument("--output", type=Path, default=default_dir / "g1_29dof.xml")
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

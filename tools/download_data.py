"""Download fixed upstream G1/LAFAN1 data and record every input hash."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
import numpy as np

REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def destination(name):
    if name.startswith("g1/") and name.endswith(".csv"):
        return DATA / "lafan1_g1" / Path(name).name
    prefix = "robot_description/g1/"
    if name == prefix + "g1_29dof_rev_1_0.urdf" or name.startswith(prefix + "meshes/"):
        return DATA / "g1_description" / name.removeprefix(prefix)
    return None


def main():
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
    wanted = [(name, destination(name)) for name in files if destination(name) is not None]

    def fetch(item):
        name, target = item
        source = hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return {"source": name, "path": str(target.relative_to(ROOT)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size}

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fetch, wanted))
    clips = sorted((DATA / "lafan1_g1").glob("*.csv"))
    arrays = [np.loadtxt(p, delimiter=",") for p in clips]
    frames = sum(a.shape[0] for a in arrays)
    if len(clips) != 40 or frames != 264705 or any(a.shape[1] != 36 for a in arrays):
        raise ValueError(f"Unexpected LAFAN1 dataset: {len(clips)} clips, {frames} frames")
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "source_fps": 30,
                "clips": len(clips), "frames": frames, "files": records}
    (DATA / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(clips)} clips, {frames} frames; data/provenance.json saved", flush=True)


if __name__ == "__main__":
    main()

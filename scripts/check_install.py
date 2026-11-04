"""Check the pinned stack without launching the simulator."""

import importlib.metadata as metadata
import os
from pathlib import Path

import casadi
import cma
import torch
from acados_template import AcadosOcp

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSIONS = {"isaacsim": "5.0.0.0", "isaaclab": "0.47.1", "torch": "2.7.0+cu128", "numpy": "1.26.0"}

for package, expected in EXPECTED_VERSIONS.items():
    installed = metadata.version(package)
    if installed != expected:
        raise RuntimeError(f"{package}: expected {expected}, found {installed}")

acados_dir = Path(os.environ["ACADOS_SOURCE_DIR"])
if acados_dir != REPO_ROOT / "external" / "acados":
    raise RuntimeError(f"ACADOS_SOURCE_DIR points to {acados_dir}. Run: source env.sh")
for required in ("lib/libacados.so", "lib/libhpipm.so", "lib/libblasfeo.so", "bin/t_renderer"):
    if not (acados_dir / required).exists():
        raise FileNotFoundError(acados_dir / required)

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU")

print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
print(f"casadi {casadi.__version__}, cma {cma.__version__}, acados at {acados_dir}")
print("Installation OK")

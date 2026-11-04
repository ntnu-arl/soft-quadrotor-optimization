#!/usr/bin/env bash
# Installs the exact software stack into the active Python 3.11 environment:
# pinned Python packages, IsaacLab and acados at fixed commits (under external/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/external"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="1103a0f38f0ca480ff739944cf829c989f5cb8cf"
ACADOS_REPO="https://github.com/acados/acados.git"
ACADOS_COMMIT="a312651abe314cc693bfc5c6c463a94839718d49"
TERA_URL="https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64"
TERA_SHA256="1973ddd1b536dcd4a059a22f7259c3b0e6b6c878cbfbe7e9b962eba195f361d9"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

die() { echo "[install] ERROR: $*" >&2; exit 1; }
step() { echo; echo "[install] $*"; }

for tool in git cmake make gcc g++ curl ffmpeg sha256sum; do
    command -v "${tool}" >/dev/null || die "'${tool}' not found. See the system packages in README.md."
done
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    die "activate a Python 3.11 conda or venv environment first"
fi
[ "${CONDA_DEFAULT_ENV:-}" != "base" ] || die "refusing to install into the conda base environment"
python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' \
    || die "Python 3.11 is required, found $(python --version)"

step "[1/4] Python packaging tools"
python -m pip install 'pip==25.2' 'setuptools==78.1.1' 'wheel==0.45.1'

step "[2/4] Pinned Python packages (Isaac Sim is a download of about 15 GB)"
python -m pip install --no-build-isolation --no-deps 'flatdict==4.0.1'
python -m pip install --no-deps --extra-index-url "${TORCH_INDEX}" -r "${REPO_ROOT}/requirements.txt"

clone_at_commit() {
    local url="$1" commit="$2" target="$3"
    if [ ! -d "${target}/.git" ]; then
        git init --quiet "${target}"
        git -C "${target}" remote add origin "${url}"
    fi
    git -C "${target}" fetch --quiet --depth 1 origin "${commit}"
    git -C "${target}" checkout --quiet --force "${commit}"
}

mkdir -p "${EXTERNAL_DIR}"

step "[3/4] IsaacLab @ ${ISAACLAB_COMMIT:0:10}"
clone_at_commit "${ISAACLAB_REPO}" "${ISAACLAB_COMMIT}" "${EXTERNAL_DIR}/IsaacLab"
for package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    python -m pip install --no-deps --editable "${EXTERNAL_DIR}/IsaacLab/source/${package}"
done

step "[4/4] acados @ ${ACADOS_COMMIT:0:10}"
ACADOS_DIR="${EXTERNAL_DIR}/acados"
clone_at_commit "${ACADOS_REPO}" "${ACADOS_COMMIT}" "${ACADOS_DIR}"
git -C "${ACADOS_DIR}" submodule update --init --depth 1 external/blasfeo external/hpipm
git -C "${ACADOS_DIR}" apply "${REPO_ROOT}/patches/acados_sim_batch_num_threads.patch"
cmake -S "${ACADOS_DIR}" -B "${ACADOS_DIR}/build" -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_OPENMP=OFF
cmake --build "${ACADOS_DIR}/build" --parallel "$(nproc)"
cmake --install "${ACADOS_DIR}/build"
python -m pip install --no-deps --editable "${ACADOS_DIR}/interfaces/acados_template"
mkdir -p "${ACADOS_DIR}/bin"
curl --fail --location --silent --show-error --output "${ACADOS_DIR}/bin/t_renderer" "${TERA_URL}"
echo "${TERA_SHA256}  ${ACADOS_DIR}/bin/t_renderer" | sha256sum --check --quiet
chmod +x "${ACADOS_DIR}/bin/t_renderer"

echo
echo "[install] Done. In every new shell run:  source env.sh"
echo "[install] Then check the installation with:  python scripts/check_install.py"

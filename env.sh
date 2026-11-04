# Source this after activating the Python environment:  source env.sh
SOFT_DRONES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACADOS_SOURCE_DIR="${SOFT_DRONES_ROOT}/external/acados"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Accepts the NVIDIA Omniverse License Agreement required by Isaac Sim:
# https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html
export OMNI_KIT_ACCEPT_EULA=YES

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --yes --no-install-recommends \
    software-properties-common gpg-agent \
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git \
    libasound2 libegl1 libgl1 libglib2.0-0 libglu1-mesa libice6 libsm6 libvulkan1 \
    vulkan-tools libx11-6 libx11-xcb1 libxcb1 libxext6 libxi6 libxkbcommon0 libxrandr2 libxrender1 libxt6 libxtst6 \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/soft_drones
COPY requirements.txt .
RUN pip install 'pip==25.2' 'setuptools==78.1.1' 'wheel==0.45.1' \
    && pip install --no-build-isolation --no-deps 'flatdict==4.0.1' \
    && pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

COPY . .
RUN ./install.sh

# Same variables as env.sh, set as image ENV so they apply to every `docker run` and `docker exec`, not only interactive shells.
ENV ACADOS_SOURCE_DIR=/workspace/soft_drones/external/acados
ENV LD_LIBRARY_PATH=/workspace/soft_drones/external/acados/lib
ENV OMNI_KIT_ACCEPT_EULA=YES
CMD ["bash"]

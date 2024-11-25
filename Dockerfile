# switched to an older version due to the old 515 driver
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# to build cubvh, we need to set the arch list
# since we are using cuda 11.7 on A100 (Ampere), we need to set the arch list to 8.0 and 8.6
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"

# system dependencies (usually do not apt-get upgrade)
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
vim git tmux htop curl wget build-essential cmake ffmpeg \
libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender1 \
python-is-python3 python3.10-dev python3-pip \
libeigen3-dev libaio-dev \
&& rm -rf /var/lib/apt/lists/*

# cv2 requires a special version (otherwise error with cv2.dnn.DictValue)
# setuptools is also a special version to build flash-attn...
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --no-cache-dir pip ninja \
&& pip install --no-cache-dir setuptools==69.5.1 \
&& pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
&& pip install --no-cache-dir git+https://github.com/ashawkey/kiuikit.git \
&& pip install --no-cache-dir jupyterlab \
&& pip install --no-cache-dir flash-attn --no-build-isolation \
&& pip install --no-cache-dir -r /tmp/requirements.txt \
&& pip install --no-cache-dir opencv-python==4.8.0.74

# dotfiles (just bashrc for now)
RUN git clone https://github.com/ashawkey/dotfiles.git /tmp/dotfiles \
&& cat /tmp/dotfiles/.bashrc >> /root/.bashrc

## build with:
# docker build -t lmm:0.1 .
# run with:
# docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ~/workspace/lmm:/projects/lmm  lmm:0.1
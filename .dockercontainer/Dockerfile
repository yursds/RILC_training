# Use the Ubuntu 22.04 base image
FROM ubuntu:22.04

# ================================== Nvidia ================================== #

# Use the NVIDIA graphics card.
ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=12.1 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,\
    driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,\
    driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,\
    driver>=510,driver<511 brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 brand=titan,\
    driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 brand=tesla,driver>=515,driver<516 brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=515,driver<516 brand=nvidiartx,\
    driver>=515,driver<516 brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 brand=titan,\
    driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,\
    driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,\
    driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"
ENV NV_CUDA_CUDART_VERSION 12.1.105-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-12-1

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 12.1.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-1=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

# ============================================================================ #

# Prevent bash to ask for user input which may break the building process
ENV DEBIAN_FRONTEND=noninteractive

# Install sudo, and some essential packages.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install --no-install-recommends -qqy \
    bash-completion \
    build-essential \
    git \
    sudo \
    wget \
    python3-pip \
    python3-tk \
    libgl1-mesa-glx

# Install pip3 packages.

# torch with constraint url
RUN --mount=type=cache,target=/root/.cache/pip \    
    pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121;

# Install Python packages
COPY config/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt

# RUN --mount=type=cache,target=/root/.cache/pip \    
#     pip3 install --no-cache-dir \
#     requests \
#     pin \
#     matplotlib \
#     tensordict \
#     meshcat_shapes \
#     gymnasium \
#     stable-baselines3[extra] \
#     mujoco \
#     imageio
    
# ============================== Change USER ================================= #
# Create the same user as the host itself. (By default Docker creates the container as root, which is not recommended.)
ARG UID=1000
ARG GID=1000
ARG USER=ros
ARG PWDR=/
RUN addgroup --gid ${GID} ${USER} \
 && adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} ${USER} \
 && usermod -a -G dialout ${USER} \
 && echo ${USER}" ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/99_aptget \
 && chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

# Choose to run as user
ENV USER ${USER}
USER ${USER}

# Change HOME environment variable
ENV HOME /home/${USER}

ENTRYPOINT [""]
CMD ["bash"]

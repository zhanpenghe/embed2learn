# NOTICE: To keep consistency across this docker file, scripts/setup_linux.sh
# and scripts/setup_macos.sh, if there's any changes applied to this file,
# specially regarding the installation of dependencies, apply those same
# changes to the mentioned files.
ARG PARENT_IMAGE=ubuntu:16.04
FROM $PARENT_IMAGE

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    wget \
    bzip2 \
    unzip \
    git \
    curl \
    # For building glfw
    cmake \
    xorg-dev \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    # 16.04 repo is too old, install glfw from source instead
    # libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    # OpenAI gym
    # See https://github.com/openai/gym/blob/master/Dockerfile
    libpq-dev \
    ffmpeg \
    libjpeg-dev \
    swig \
    libsdl2-dev \
    # OpenAI baselines
    libopenmpi-dev \
    openmpi-bin \
    python3-pip && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Build GLFW because the Ubuntu 16.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
RUN apt-get purge -y -v libglfw*
RUN git clone https://github.com/glfw/glfw.git && \
  cd glfw && \
  git checkout 0be4f3f75aebd9d24583ee86590a38e741db0904 && \
  mkdir glfw-build && \
  cd glfw-build && \
  cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. && \
  make -j"$(nproc)" && \
  make install && \
  cd ../../ && \
  rm -rf glfw

# MuJoCo 1.5 (for gym)
RUN mkdir -p /root/.mujoco && \
  wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

RUN pip3 install pipenv

# MuJoCo 2.0 (for dm_control)
RUN mkdir -p /root/.mujoco && \
  wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip && \
  ln -s $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
  ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# conda environment
# Copy over just environment.yml and setup.py first, so the Docker cache doesn't
# expire until they change
#
# Files needed to run setup.py
# - README.md
# - VERSION
# - scripts/garage
# - src/garage/__init__.py
# - setup.py
COPY Pipfile /root/code/embed2learn/Pipfile
# COPY README.md /root/code/embed2learn/README.md
# COPY VERSION /root/code/embed2learn/VERSION
# COPY scripts/garage /root/code/embed2learn/scripts/garage
# COPY src/garage/__init__.py /root/code/garage/src/garage/__init__.py
# COPY setup.py /root/code/embed2learn/setup.py
# COPY environment.yml /root/code/garage/environment.yml

# We need a MuJoCo key to install mujoco_py
# In this step only the presence of the file mjkey.txt is required, so we only
# create an empty file
ARG MJKEY
COPY mjkey.txt /root/.mujoco/mjkey.txt

RUN wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz -P /opt
RUN tar -xvf /opt/Python-3.6.3.tgz -C /opt
RUN /opt/Python-3.6.3/configure
RUN make
RUN make install

# Extras
# prevent pip from complaining about available upgrades
# RUN ["/bin/bash", "-c", "source activate garage && pip install --upgrade pip"]

# Setup repo
WORKDIR /root/code/embed2learn
# Pre-build pre-commit env
# COPY .pre-commit-config.yaml /root/code/embed2learn
RUN ["/bin/bash", "-c", "pipenv install --dev"]
# RUN ["/bin/bash", "-c", "pipenv shell"]

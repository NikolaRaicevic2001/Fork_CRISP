# Base image
FROM ubuntu:22.04

# Avoid interactive tzdata prompts, etc.
ENV DEBIAN_FRONTEND=noninteractive

# ===========================================
# System deps (build tools, C++ libs, Python)
# ===========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python-is-python3 \
    python3-tk \
    python3-pybind11 \
    libeigen3-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libyaml-dev \
    # --- Qt/X11/GL deps ---
    libx11-xcb1 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-xfixes0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrender1 \
    libxi6 \
    libsm6 \
    libfontconfig1 \
    libgl1 \
    libegl1 \
    libopengl0 \
    libdbus-1-3 \
    libglib2.0-0 \
    qtbase5-dev \
    qtwayland5 \
    python3-pyqt5 \
    # --- helpful utilities ---
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ===========================================
# Python deps
# ===========================================
# Upgrade build tooling (helps with manylinux wheels)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin NumPy 1.x so old Matplotlib wheels load correctly
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    matplotlib==3.5.3 \
    pandas

# ===========================================
# App workspace
# ===========================================
WORKDIR /app

# Copy project sources
COPY ./src ./src

# ===========================================
# Third-party C++ deps (CppAD, CppADCodeGen, piqp)
# ===========================================
# CppAD
RUN mkdir -p src/third_party/CppAD/build && \
    cd src/third_party/CppAD/build && \
    cmake .. && \
    make -j"$(nproc)" install

# CppADCodeGen
RUN mkdir -p src/third_party/CppADCodeGen/build && \
    cd src/third_party/CppADCodeGen/build && \
    cmake .. && \
    make -j"$(nproc)" install

# piqp
RUN mkdir -p src/third_party/piqp/build && \
    cd src/third_party/piqp/build && \
    cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF && \
    cmake --build . --config Release -j"$(nproc)" && \
    cmake --install . --config Release
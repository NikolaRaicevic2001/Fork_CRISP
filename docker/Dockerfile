FROM ubuntu:22.04

# Suppress all interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt update && apt install -y \
    libboost-all-dev \
    libyaml-cpp-dev \
    build-essential \
    libeigen3-dev \
    libyaml-dev \
    pkg-config \
    cmake \
    python-is-python3 \
    python3-pybind11 \
    python3-pip \
    python3-tk \
    python3 \
    git \
    && apt clean

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir \
    matplotlib==3.5.3 \
    pandas \
    pyqt5

# Set the working directory
WORKDIR /app

# Copy the project files into the image
COPY ./src ./src

# Install CppAD Dependencies
RUN mkdir -p src/third_party/CppAD/build && \
    cd src/third_party/CppAD/build && \
    cmake .. && make -j$(nproc) install

# Install CppADCodeGen Dependencies
RUN mkdir -p src/third_party/CppADCodeGen/build && \
    cd src/third_party/CppADCodeGen/build && \
    cmake .. && make -j$(nproc) install

# Install piqp Dependencies
RUN mkdir -p src/third_party/piqp/build && \
    cd src/third_party/piqp/build && \
    cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF && \
    cmake --build . --config Release -j$(nproc) && \
    cmake --install . --config Release



# # Missing Qt dependencies for Pushbox visualization
# apt install sudo
# sudo apt-get update
# sudo apt-get install -y \
#   libx11-xcb1 libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
#   libxcb-render0 libxcb-render-util0 libxcb-shm0 libxcb-randr0 libxcb-xfixes0 \
#   libxkbcommon0 libxkbcommon-x11-0 libxrender1 libxi6 libsm6 libfontconfig1 \
#   libgl1 libegl1 libopengl0 libdbus-1-3 libglib2.0-0 qtbase5-dev \
#   python3-pyqt5 qtwayland5

  
FROM ubuntu:22.04

# Update and install dependencies
RUN apt update && apt install -y \
    cmake \
    libeigen3-dev \
    libboost-all-dev \
    libyaml-dev \
    libyaml-cpp-dev \
    python3 \
    python3-pybind11 \
    python-is-python3 \
    pkg-config \
    build-essential \
    git \
    && apt clean

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

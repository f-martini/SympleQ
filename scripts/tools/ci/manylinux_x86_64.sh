#!/usr/bin/env bash
set -e

# Install EPEL and required tools
if [ -z "$GCC_VERSION" ]; then
    if [ -n "$1" ]; then
        GCC_VERSION="$1"
    else
        echo "Error: GCC_VERSION not set. Please provide as env var or first argument."
        exit 1
    fi
fi

if [ -z "$CUDA_VERSION" ]; then
    if [ -n "$2" ]; then
        CUDA_VERSION="$2"
    else
        echo "Error: CUDA_VERSION not set. Please provide as env var or second argument."
        exit 1
    fi
fi

if [ -z "$VCPKG_COMMIT_ID" ]; then
    if [ -n "$3" ]; then
        VCPKG_COMMIT_ID="$3"
    else
        echo "Error: VCPKG_COMMIT_ID not set. Please provide as env var or third argument."
        exit 1
    fi
fi

yum install -y epel-release
yum remove -y gcc gcc-c++
yum install -y make git which gcc-toolset-${GCC_VERSION} curl zip unzip tar
source /opt/rh/gcc-toolset-${GCC_VERSION}/enable

echo "GCC version:"
gcc --version
echo "GCC path:"
which gcc

# Install CUDA
echo "Installing CUDA..."
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}.0/local_installers/cuda_${CUDA_VERSION}.0_555.42.02_linux.run" -o cuda_installer.run
sh cuda_installer.run --silent --toolkit --no-opengl-libs --no-man-page --no-drm
ln -sf "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

# Install vcpkg
echo "Installing vcpkg..."
git clone https://github.com/Microsoft/vcpkg.git /tmp/vcpkg
cd /tmp/vcpkg
git checkout "${VCPKG_COMMIT_ID}"
./bootstrap-vcpkg.sh

echo "CUDA and vcpkg installation complete"
export PATH=/usr/local/cuda/bin:/opt/rh/gcc-toolset-12/root/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++\


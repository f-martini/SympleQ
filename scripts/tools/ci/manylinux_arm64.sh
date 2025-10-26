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

# Install CUDA for ARM64
echo "Installing CUDA for ARM64..."
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/aarch64/cuda-rhel8.repo -o /etc/yum.repos.d/cuda-rhel8.repo
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/aarch64/D42D0685.pub | rpm --import -
yum clean all
yum makecache
CUDA_PKG_VERSION=$(echo "${CUDA_VERSION}" | tr '.' '-')
yum install -y "cuda-toolkit-${CUDA_PKG_VERSION}"
ln -sf "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

echo "CUDA ${CUDA_VERSION} installation complete"

# Install vcpkg
echo "Installing vcpkg..."
git clone https://github.com/Microsoft/vcpkg.git /tmp/vcpkg
cd /tmp/vcpkg
git checkout "${VCPKG_COMMIT_ID}"
./bootstrap-vcpkg.sh

echo "CUDA and vcpkg installation complete"
export PATH=/usr/local/cuda/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CC=/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin/g++

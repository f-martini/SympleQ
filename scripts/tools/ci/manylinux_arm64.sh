#!/usr/bin/env bash
set -e

# Install EPEL and required tools
yum install -y epel-release
yum remove -y gcc gcc-c++
yum install -y make git which gcc-toolset-${GCC_VERSION} curl zip unzip tar

# Enable GCC toolset
source /opt/rh/gcc-toolset-${GCC_VERSION}/enable

echo "GCC version:"
gcc --version
echo "GCC path:"
which gcc

# Install CUDA for ARM64
echo "Installing CUDA for ARM64..."
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/aarch64/cuda-repo-rhel8-${CUDA_VERSION}-local-${CUDA_VERSION}.0.0-1.aarch64.rpm" -o cuda-repo.rpm
yum install -y ./cuda-repo.rpm
yum clean all
yum install -y cuda-toolkit-${CUDA_VERSION}
ln -sf "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

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

# GitHub Actions for QuAOS Project

This directory contains reusable GitHub Actions for building and publishing Python wheels with C++/CUDA dependencies.

## Actions Overview

### Setup Actions

#### `setup/python-environment`
Sets up Python environment with build dependencies.

**Inputs:**
- `python-version` (required): Python version to setup
- `architecture` (optional): Architecture (default: 'x64')
- `include-cibuildwheel` (optional): Install cibuildwheel (default: 'true')

**Outputs:**
- `cibw-build`: Generated CIBW_BUILD value

**Usage:**
```yaml
- name: Setup Python environment
  id: python-setup
  uses: ./.github/actions/setup/python-environment
  with:
    python-version: '3.11'
    architecture: 'x64'
```

#### `setup/cuda`
Sets up CUDA toolkit for Linux and Windows.

**Inputs:**
- `cuda-version` (required): CUDA version to install
- `platform` (required): Target platform (linux or windows)
- `method` (optional): Installation method (default: 'network')

**Outputs:**
- `cuda-path`: Path to CUDA installation

#### `setup/vcpkg`
Sets up vcpkg package manager.

**Inputs:**
- `vcpkg-commit` (required): vcpkg commit ID to checkout
- `platform` (required): Target platform
- `install-path` (optional): Installation path (default: '/tmp/vcpkg')

**Outputs:**
- `vcpkg-root`: Path to vcpkg root
- `cmake-toolchain`: Path to CMake toolchain file

### Build Actions

#### `build/wheel`
Builds Python wheels using cibuildwheel or direct build.

**Inputs:**
- `platform` (required): Target platform
- `architecture` (required): Target architecture
- `python-version` (required): Python version
- `cibw-build` (required): CIBW_BUILD value
- `cmake-preset` (required): CMake preset to use
- Plus many optional inputs for CUDA, vcpkg, etc.

#### `build/upload-artifacts`
Uploads built wheels as GitHub artifacts.

**Inputs:**
- `artifact-name` (required): Name for the artifact
- `wheel-path` (optional): Path to wheels (default: 'dist/*.whl')
- `retention-days` (optional): Retention period (default: '30')

#### `build/publish`
Downloads artifacts and publishes wheels to PyPI.

**Inputs:**
- `project-name` (required): Project name for filtering
- `test-pypi-token` (optional): Test PyPI token
- `pypi-token` (optional): Production PyPI token
- `publish-to-test` (optional): Publish to Test PyPI
- `publish-to-pypi` (optional): Publish to PyPI

### Cleanup Actions

#### `cleanup/free-disk-space`
Frees up disk space on GitHub runners.

**Inputs:**
- `platform` (required): Target platform (linux or windows)
- `aggressive` (optional): Perform aggressive cleanup (default: 'false')

## Example Workflow

```yaml
name: Build Wheels
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12", "3.13"]
    
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Free up disk space
        uses: ./.github/actions/cleanup/free-disk-space
        with:
          platform: linux
          aggressive: 'true'

      - name: Setup Python
        id: python-setup
        uses: ./.github/actions/setup/python-environment
        with:
          python-version: ${{ matrix.python }}

      - name: Setup CUDA
        id: cuda-setup
        uses: ./.github/actions/setup/cuda
        with:
          cuda-version: '12.6'
          platform: linux

      - name: Setup vcpkg
        id: vcpkg-setup
        uses: ./.github/actions/setup/vcpkg
        with:
          vcpkg-commit: '15fd3bbcf7cc66249ba11f87a6215ee6f291bb26'
          platform: linux

      - name: Build wheels
        uses: ./.github/actions/build/wheel
        with:
          platform: linux
          architecture: x86_64
          python-version: ${{ matrix.python }}
          cibw-build: ${{ steps.python-setup.outputs.cibw-build }}
          cuda-path: ${{ steps.cuda-setup.outputs.cuda-path }}
          vcpkg-root: ${{ steps.vcpkg-setup.outputs.vcpkg-root }}
          cmake-toolchain: ${{ steps.vcpkg-setup.outputs.cmake-toolchain }}
          cmake-preset: linux-x86_64-Release

      - name: Upload artifacts
        uses: ./.github/actions/build/upload-artifacts
        with:
          artifact-name: wheels-linux-py${{ matrix.python }}
```

## Benefits of This Refactoring

1. **Reusability**: Actions can be used across multiple workflows
2. **Maintainability**: Changes to build logic only need to be made in one place
3. **Testability**: Individual actions can be tested separately
4. **Modularity**: Each action has a single responsibility
5. **Documentation**: Clear inputs/outputs for each action
6. **Error Handling**: Consistent error handling across actions
7. **Consistency**: Standardized approach to similar tasks

## Migration Guide

The original workflow steps have been replaced with calls to these actions:

| Original Step | New Action |
|--------------|------------|
| Free up disk space | `cleanup/free-disk-space` |
| Set up Python + pip installs | `setup/python-environment` |
| CUDA installation | `setup/cuda` |
| vcpkg setup | `setup/vcpkg` |
| cibuildwheel build | `build/wheel` |
| Upload artifacts | `build/upload-artifacts` |
| PyPI publishing | `build/publish` |

The workflow is now much cleaner and easier to maintain!
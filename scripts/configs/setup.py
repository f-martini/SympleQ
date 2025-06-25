from setuptools import setup, find_packages
from pathlib import Path
import subprocess

current_file_path = Path(__file__).parent

dependencies_path = current_file_path / 'requirements.txt'
dependencies = [line.strip() for line in open(dependencies_path) if line.strip() and not line.startswith('#')]

def get_version():
    try:
        version = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            cwd=current_file_path / "../..",
            encoding='utf-8'
        ).strip()
        return version
    except Exception:
        return "0.0.0.0"

setup(
    name='quaos',
    version=get_version(),
    description='QuAOS: Quantum Algorithms, Optomechanics and Simulations',
    long_description=(current_file_path / '../..' / 'README.md').read_text(),
    package_dir={'': '../../src'},
    packages=find_packages(where="../../src", exclude=['tests']),
    install_requires=dependencies,
    author='Quaos-Lab',
)

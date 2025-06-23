from setuptools import setup, find_packages
from pathlib import Path
current_file_path = Path(__file__).parent

dependencies_path = current_file_path / 'requirements.txt'
dependencies = [line.strip() for line in open(dependencies_path) if line.strip() and not line.startswith('#')]

setup(
    name='quaos',
    version='0.1',
    description='Quaos: Quantum Algorithms for Optimization and Simulation',
    long_description=(current_file_path / '..' / 'README.md').read_text(),
    package_dir={'': '../src'},
    packages=find_packages(where="../src", exclude=['tests']),
    install_requires=dependencies,
    author='Quaos-Lab',
)

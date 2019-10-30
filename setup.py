from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "gym",
    "tensorflow==2.0.0",
    "tensorflow-probability==0.8.0",
    "attrs",
    "numpy",
    "pandas",
    "matplotlib",
]

setup(
    name="snake",
    version="0.0.0",
    url="https://github.com/fomorians/snake",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)

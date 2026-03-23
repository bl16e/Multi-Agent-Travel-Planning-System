from setuptools import setup, find_packages

setup(
    name="masystem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
)

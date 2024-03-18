from setuptools import setup, find_packages

setup(name="goat",
      version='1.0',
      description='Learning gene embedding of multi-omics regulation using graph neural network',
      packages=find_packages(include=['goat']),
      install_requires=['pandas', 'PyYAML']
)

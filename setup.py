from setuptools import setup, find_packages

from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='veripupil',
      version='0.0.11',
      description='Python classes for optimizing models against veri',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/bgokden/modeloptimizer',
      author='Berk Gokden',
      author_email='berkgokden@gmail.com',
      license='MIT',
      keywords='veri service python client ml keras tensorflow',
      packages=find_packages(exclude=['tests*']),
      install_requires=['veriservice', 'keras', 'tensorflow'])

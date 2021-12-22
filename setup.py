from setuptools import setup
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="DeepPurpose", 
    packages = ['DeepPurpose'],
    package_data={'DeepPurpose': ['ESPF/*']},
    version="0.1.5",
    author="Kexin Huang, Tianfan Fu",
    license="BSD-3-Clause",
    author_email="kexinhuang@hsph.harvard.edu",
    description="a Deep Learning Based Toolkit for Drug Discovery",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kexinhuang12345/DeepPurpose",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)

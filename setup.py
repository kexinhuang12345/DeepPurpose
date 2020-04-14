import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepPurpose", 
    version="0.0.1",
    author="Kexin Huang, Tianfan Fu",
    author_email="kexinhuang@hsph.harvard.edu",
    description="a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kexinhuang12345/DeepPurpose",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Zero v1.0 Universal",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.7',
)
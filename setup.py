import setuptools
from hugdatafast.__init__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PKGS = [
    'fastai',
    'nlp',
]

setuptools.setup(
    name="hugdatafast",
    version=__version__,
    author="Richard Wang",
    author_email="richardyy1188@gmail.com",
    description="The elegant bridge between hugginface data and fastai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richarddwang/hugdatafast",
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    install_requires=REQUIRED_PKGS,
    keywords='nlp machine learning datasets metrics fastai huggingface',
)
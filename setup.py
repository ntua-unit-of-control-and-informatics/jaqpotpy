from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jaqpotpy",
    version="{{VERSION_PLACEHOLDER}}",
    description="Client library for managing machine learning models on the Jaqpot platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy",
    author="Unit of Process Control and Informatics | National Technical University of Athens",
    author_email="upci.ntua@gmail.com",
    license="MIT License",
    packages=find_packages(exclude=["*.tests"]),
    package_data={"jaqpotpy": ["data/*.gz"]},
    install_requires=[
        "pandas==2.2.2",
        "pyjwt==2.8.0",
        "simplejson==3.19.2",
        "pydotplus==2.0.2",
        "requests==2.32.2",
        "pydantic==2.7.1",
        "rdkit==2023.9.6",
        "mordredcommunity==2.0.5",
        "scikit-learn==1.5.0",
        "tqdm==4.66.4",
        "kennard-stone==2.2.1",
        "mendeleev==0.16.2",
        "pymatgen==2024.5.1",
        "skl2onnx==1.17.0",
        "onnx==1.17.0",
        "onnxruntime==1.19.2",
        "httpx==0.27.0",
        "attrs==23.2.0",
        "python-keycloak==4.3.0",
        "ruff==0.6.3",
        "polling2==0.5.0",
        "python-dotenv==1.0.1",
        "torch==2.3.0",
        "torch-geometric==2.5.0",
        "onnxmltools==1.12.0",
        "xgboost==2.1.1",
        "pre-commit==4.0.1",
    ],
    dependency_links=["https://data.pyg.org/whl/torch-2.3.0+cpu.html"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

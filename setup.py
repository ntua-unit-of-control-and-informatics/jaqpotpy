from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(name='jaqpotpy',
      version='{{VERSION_PLACEHOLDER}}',
      description='Standardizing molecular modeling',
      long_description=long_description,
      long_description_content_type='text/markdown',

      url='https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy',
      author='Unit of Process Control and Informatics | National Technical University of Athens',
      author_email='upci.ntua@gmail.com',
      license='MIT License',
      packages=find_packages(exclude=["*.tests"]),
      package_data={'jaqpotpy': ['data/*.gz']},
      install_requires=[
          'pandas', 'requests', 'pydantic', 'rdkit-pypi', 'mordred', 'pyjwt', 'scikit-learn', 'tqdm',
          'skl2onnx', 'onnxruntime'
      ],
      zip_safe=False)

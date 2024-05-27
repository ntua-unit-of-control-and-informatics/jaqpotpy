from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

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

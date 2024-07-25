from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='jaqpotpy',
      version= '{{VERSION_PLACEHOLDER}}',
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
            'pandas==2.2.2',
            'pyjwt==2.8.0',
            'simplejson==3.19.2',
            'pydotplus==2.0.2',
            'requests==2.32.2',
            'pydantic==2.7.1',
            'rdkit==2023.9.6',
            'mordredcommunity==2.0.5',
            'scikit-learn==1.5.0',
            'tqdm==4.66.4',
            'kennard-stone==2.2.1',
            'mendeleev==0.16.2',
            'pymatgen==2024.5.1',
            'skl2onnx==1.16.0',
            'onnxruntime==1.18.0',
            'httpx==0.27.0',
            'attrs==23.2.0',
            'compress_pickle==2.1.0',
            'torch==2.3.0',
            'torch-geometric==2.3.1',
            'torchvision==0.18.0',
      ],
      zip_safe=False)

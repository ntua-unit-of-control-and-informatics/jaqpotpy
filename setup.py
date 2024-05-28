from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# python setup.py bdist_wheel
# twine upload dist/*
# twine upload --repository testpypi dist/jaqpotpy-2.0.0b0-py3-none-any.whl
# twine upload dist/jaqpotpy-1.0.84-py3-none-any.whl
# docker build -t euclia/jaqpotpy:1.0.3 --no-cache python setup.py bdist_wheel--build-arg tag=1.0.3 .


version = '1.0.101'

setup(name='jaqpotpy',
      # version='2.0.5-beta0',
      version='1.0.101',
      description='Standardizing molecular modeling',
      long_description=long_description,
      long_description_content_type='text/markdown',

      url='https://github.com/euclia/jaqpotpy',
      author='Pantelis Karatzas',
      author_email='pantelispanka@gmail.com',
      license='MIT License',
      packages=find_packages(exclude=["*.tests"]),
      package_data={'jaqpotpy': ['data/*.gz']},
      # package_data={find_packages(exclude=["*.tests"]): ['data/*.gz']},
      # packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers', 'jaqpotpy.utils'
      #           , 'jaqpotpy.entities', 'jaqpotpy.dto', 'jaqpotpy.doa'
      #           , 'jaqpotpy.helpers', 'jaqpotpy.colorlog', 'jaqpotpy.models', 'jaqpotpy.cfg'
      #           , 'jaqpotpy.datasets', 'jaqpotpy.descriptors', 'jaqpotpy.descriptors.molecular'
      #           , 'jaqpotpy.descriptors.graph'],
      install_requires=[
      #     'pandas', 'requests', 'pydantic', 'rdkit-pypi', 'mordred', 'pyjwt', 'scikit-learn', 
      #     'tqdm', 'skl2onnx', 'onnxruntime'
           
            'pandas==2.2.2',
            'pyjwt==2.8.0',
            'simplejson==3.19.2',
            'pydotplus==2.0.2',
            'requests==2.32.2',
            'pydantic==2.7.1',
            'rdkit-pypi==2022.9.5',
            'mordred==1.2.0',
            'scikit-learn==1.5.0',
            'tqdm==4.66.4',
            'kennard-stone==2.2.1',
            'mendeleev==0.16.2',
            'pymatgen==2024.5.1',
            'skl2onnx==1.16.0',
            'onnxruntime==1.18.0',

            # # Torch requirements
            'torch==2.3.0',
            'torch-geometric==2.3.1',
            'torchvision==0.18.0',
            # 'torch-scatter==2.1.2',
            # 'torch-sparse==0.6.18',
            # 'torch-cluster==1.6.3',
            # 'torch-spline-conv==1.2.2'
            # # ,'tornado', 'scikit-learn', 'torch', 'torch-scatter', 'torch-sparse', 'torch-cluster'
            # # , 'torch-spline-conv', 'torch-geometric', 'dgl', 'torch_sparse', 'kennard-stone'
      ],
      zip_safe=False)

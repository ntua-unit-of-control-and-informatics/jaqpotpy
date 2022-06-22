from setuptools import setup, find_packages

# python setup.py bdist_wheel
# twine upload dist/*
# twine upload --repository testpypi dist/jaqpotpy-2.0.0b0-py3-none-any.whl
<<<<<<< Updated upstream
# twine upload dist/jaqpotpy-1.0.34-py3-none-any.whl
# docker build -t euclia/jaqpotpy:1.0.3 --no-cache python setup.py bdist_wheel--build-arg tag=1.0.3 .

setup(name='jaqpotpy',
      # version='2.0.5-beta0',
      version='1.0.47',
=======
# twine upload dist/jaqpotpy-1.0.48-py3-none-any.whl
# docker build -t euclia/jaqpotpy:1.0.3 --no-cache --build-arg tag=1.0.3 .

setup(name='jaqpotpy',
      # version='2.0.5-beta0',
      version='1.0.48',
>>>>>>> Stashed changes
      description='Python client for Jaqpot',
      url='https://github.com/euclia/jaqpotpy',
      author='Pantelis Karatzas',
      author_email='pantelispanka@gmail.com',
      license='MIT License',
      packages=find_packages(exclude=["*.tests"]),
      # packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers', 'jaqpotpy.utils'
      #           , 'jaqpotpy.entities', 'jaqpotpy.dto', 'jaqpotpy.doa'
      #           , 'jaqpotpy.helpers', 'jaqpotpy.colorlog', 'jaqpotpy.models', 'jaqpotpy.cfg'
      #           , 'jaqpotpy.datasets', 'jaqpotpy.descriptors', 'jaqpotpy.descriptors.molecular'
      #           , 'jaqpotpy.descriptors.graph'],
      install_requires=[
            'pandas', 'requests', 'pydantic', 'rdkit-pypi', 'mordred', 'pyjwt', 'scikit-learn'
            # ,'tornado',
            # 'scikit-learn', 'torch', 'torch-scatter', 'torch-sparse', 'torch-cluster'
            # , 'torch-spline-conv', 'torch-geometric', 'dgl', 'torch_sparse', 'kennard-stone'
      ],
      zip_safe=False)

# .sdf
#
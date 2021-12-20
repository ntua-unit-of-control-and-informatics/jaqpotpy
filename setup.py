from setuptools import setup

# python setup.py bdist_wheel
# twine upload dist/*

setup(name='jaqpotpy',
      version='1.0.16',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers'
                , 'jaqpotpy.entities', 'jaqpotpy.dto'
                , 'jaqpotpy.helpers', 'jaqpotpy.colorlog', 'jaqpotpy.models'],
      install_requires=[
            'pandas', 'pyjwt', 'requests', 'pydantic', 'tornado', 'rdkit-pypi'
            , 'mordred', 'scikit-learn', 'torch', 'torch-scatter', '', 'torch-sparse', 'torch-cluster'
            , 'torch-spline-conv', 'torch-geometric', 'dgl', 'torch_sparse', 'kennard-stone'
      ],
      zip_safe=False)

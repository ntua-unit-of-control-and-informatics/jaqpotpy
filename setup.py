from setuptools import setup

# python setup.py bdist_wheel
# twine upload dist/*

setup(name='jaqpotpy',
      version='1.0.11',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers'
                , 'jaqpotpy.entities', 'jaqpotpy.dto'
                , 'jaqpotpy.helpers', 'jaqpotpy.colorlog', 'jaqpotpy.models'],
      install_requires=[
            'pandas', 'pyjwt', 'requests', 'pydantic'
      ],
      zip_safe=False)

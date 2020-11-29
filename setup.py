from setuptools import setup

setup(name='jaqpotpy',
      version='0.0.4',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers'
                , 'jaqpotpy.entities', 'jaqpotpy.dto'
                , 'jaqpotpy.helpers', 'jaqpotpy.colorlog'],
      install_requires=[
            'pandas', 'pyjwt', 'requests'
      ],
      zip_safe=False)

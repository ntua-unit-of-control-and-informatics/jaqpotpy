from setuptools import setup

setup(name='jaqpotpy',
      version='0.0.1',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas, Angleos Valsamis, Pantelis Sopasakis',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers', 'jaqpotpy.entities', 'jaqpotpy.dto', 'jaqpotpy.helpers'],
      install_requires=[
            'tornado==4.5.3', 'pandas', 'pyjwt', 'requests'
      ],
      zip_safe=False)

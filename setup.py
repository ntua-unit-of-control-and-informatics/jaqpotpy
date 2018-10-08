from setuptools import setup

setup(name='jaqpotpy',
      version='0.0.1',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas, Pantelis Sopasakis, Angleos Valsamis',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.api', 'jaqpotpy.mappers', 'jaqpotpy.entities', 'jaqpotpy.dto'],
      install_requires=[
            'tornado', 'pandas'
      ],
      zip_safe=False)

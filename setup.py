from setuptools import setup

setup(name='jaqpotpy',
      version='0.1',
      description='Python client for Jaqpot',
      url='https://github.com/KinkyDesign/jaqpotpy',
      author='Pantelis Karatzas, Angleos Valsamis',
      author_email='pantelispanka@gmail.com',
      license='GNU General Public License v3.0',
      packages=['jaqpotpy', 'jaqpotpy.login'],
      install_requires=[
            'tornado'
      ],
      zip_safe=False)

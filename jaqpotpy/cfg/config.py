import os
import jaqpotpy
"""
Configuration variables for Jaqpotpy
"""

"""
x -> Test variable
verbode -> Verbose outputs. eg descriptor creation, model training
global_seed -> Seed for various descriptors and models
version -> Jaqpotpy version
"""


x = 0
verbose = True
global_seed = 42
# version = jaqpotpy.__version__
try:
    jaqpotpy_docker = os.environ['JAQPOTPY_DOCKER']
except KeyError as e:
    jaqpotpy_docker = None

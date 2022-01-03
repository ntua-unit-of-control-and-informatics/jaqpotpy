import unittest
from jaqpotpy.cfg import config


class TestConfig(unittest.TestCase):

    def test_config_1(self):
        assert config.x == 0

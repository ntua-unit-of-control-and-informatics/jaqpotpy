import unittest
from jaqpotpy.cfg import config

@unittest.skip("Not clear what this test was supposed to do")
class TestConfig(unittest.TestCase):

    def test_config_1(self):
        assert config.x == 0

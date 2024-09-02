import contextlib
import io
import sys
import unittest
import os
from datetime import datetime


@contextlib.contextmanager
def err_to(file):
    old_err = sys.stderr
    sys.stderr = file
    yield
    sys.stderr = old_err


if __name__ == "__main__":
    os.chdir("./run_tests")

    loader = unittest.TestLoader()
    start_dir = "../"
    suite = loader.discover(start_dir)

    result = io.StringIO()
    with err_to(result):
        runner = unittest.TextTestRunner()
        runner.run(suite)
        # unittest.main(exit=False)
    result.seek(0)

    # delete created files
    files = [
        item
        for item in os.listdir()
        if item != "run_tests.py"
        and item != "__init__.py"
        and item != "__pycache__"
        and "test_logs" not in item
    ]

    for item in files:
        os.remove(item)

    timestamp = datetime.now()
    with open(
        f"./test_logs-{timestamp}".replace(":", ".").replace(" ", "-"), "w"
    ) as logs:
        logs.write(result.read())

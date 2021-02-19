
import distutils.core

def pytest_configure(config):
    distutils.core.run_setup('./setup.py', script_args=['build'], stop_after='run')
    return

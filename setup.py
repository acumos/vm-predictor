# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

# extract __version__ from version file. importing will lead to install failures
setup_dir = os.path.dirname(__file__)
with open(os.path.join(setup_dir, 'vm_predictor', '_version.py')) as file:
    globals_dict = dict()
    exec(file.read(), globals_dict)
    __version__ = globals_dict['__version__']


setup(
    name = "vm_predictor",
    version = __version__,
    packages = find_packages(),
    author = "Michael Tinnemeier",
    author_email = "ezavesky@research.att.com",
    description = ("VM resource predictor based on historical data and context"),
    long_description = ("VM resource predictor based on historical data and context"),
    license = "Apache",
    package_data={},
    scripts=['bin/run_vm-predictor_reference.py'],
    entry_points="""
    [console_scripts]
    """,
    #setup_requires=['pytest-runner'],
    install_requires=['asimov-client',
                      'numpy',
                      'scipy',
                      'sklearn'],
    tests_require=['pytest',
                   'sklearn',
                   'numpy',
                   'scipy',
                   'pexpect'],
    include_package_data=True,
    )

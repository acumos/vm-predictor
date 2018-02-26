# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# extract __version__ from version file. importing will lead to install failures
setup_dir = os.path.dirname(__file__)
with open(os.path.join(setup_dir, 'vm_predictor', '_version.py')) as file:
    globals_dict = dict()
    exec(file.read(), globals_dict)
    __version__ = globals_dict['__version__']
    __model_name__ = globals_dict['__model_name__']


# warning (if run in verbose mode) about installing this object
class new_install(install):
    def run(self):
        with open(os.path.join(setup_dir, __model_name__, "INSTALL.txt")) as f:
            print(f.read())
        install.run(self)


# read requirements list from supplementary file in this repo
# requirement_list = [line for line in open(os.path.join(setup_dir, 'requirements.txt')) if line and line[0] != '#']


setup(
    name=__model_name__,
    version=__version__,
    packages=find_packages(),  # NOTE - THIS SHOULD NOT BE AN INSTALLED PACKAGE!
    author="Michael Tinnemeier",
    author_email="ezavesky@research.att.com",
    description=("VM resource predictor based on historical data and context"),
    long_description=("VM resource predictor based on historical data and context"),
    license="Apache",
    package_data={__model_name__: ['requirements.txt', 'INSTALL.txt']},
    # setup_requires=['pytest-runner'],
    entry_points="""
    [console_scripts]
    """,
    # setup_requires=['pytest-runner'],
    install_requires=[
        "acumos",
        "numpy",
        "sklearn",
        "pandas",
        "matplotlib"
    ],
    tests_require=[],
    cmdclass={'install': new_install},
    include_package_data=True,
)

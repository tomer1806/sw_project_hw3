#Setup script for the symnmf module
from setuptools import Extension, setup
module = Extension('symnmf',sources=['symnmfmodule.c', 'symnmf.c'],)
setup(
    name='symnmf',
    version='1.0',
    description='Symnmf C extension Python wrapper',
    ext_modules=[module]
)
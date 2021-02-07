from distutils.core import setup, Extension
 
module = Extension('tibs_code_generator', sources = ['code-generator.cpp'])
 
setup(
    name = 'Tibs code generator',
    version = '0.1',
    description = 'TODO fill this in',
    ext_modules = [module]
)

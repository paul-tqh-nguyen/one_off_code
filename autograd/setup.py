from setuptools import setup, find_packages

setup(
    name='autograd',
    description='Automatic Differentiation Engine',
    author='Paul Nguyen',
    author_email='paul.tqh.nguyen@gmail.com',
    version='0.0.1',
    packages=find_packages(include=['autograd']),
    python_requires='>=3.7',
    install_requires=['numpy', 'more-itertools', 'forbiddenfruit'],
    classifiers=['Programming Language :: Python :: 3 :: Only'],
    # url='' # @todo fill this in
)

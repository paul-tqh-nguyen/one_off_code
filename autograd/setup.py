from setuptools import setup, find_packages

setup(
    name="autograd",
    description="Automatic Differentiation Engine",
    author="Paul Nguyen",
    author_email="paul.tqh.nguyen@gmail.com",
    packages=find_packages(include=["metagraph", "metagraph.*"]),
    python_requires=">=3.7",
    install_requires=["numpy", "more-itertools", "forbiddenfruit"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)

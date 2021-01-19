from setuptools import setup, find_packages

setup(
    name="leibniz",
    description="A domain-specific language for automatic differentiation via static computation graphs.",
    author="Paul Nguyen",
    packages=["leibniz"],
    python_requires=">=3.7",
)

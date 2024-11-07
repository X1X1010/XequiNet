from setuptools import setup

setup(
    name="XequiNet",
    version="1.1",
    packages=["xequinet"],
    entry_points={
        "console_scripts": ["xeq = xequinet.main:main"],
    },
)

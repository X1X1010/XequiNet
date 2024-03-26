from setuptools import setup
setup(
    name = "XequiNet",
    version = "0.4.1",
    packages = ["xequinet"],
    entry_points={
        'console_scripts': [
            "xeq = xequinet.main:main",
        ]
    }
)
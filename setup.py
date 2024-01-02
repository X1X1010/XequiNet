from setuptools import setup
setup(
    name = "XequiNet",
    version = "0.3.0",
    packages = ["xequinet"],
    entry_points={
        'console_scripts': [
            "xeqtrain = xequinet.train:main",
            "xeqjit = xequinet.jit_script:main",
            "xeqinfer = xequinet.inference:main",
            "xeqtest = xequinet.test:main",
            "xeqopt = xequinet.geo_opt:main"
        ]
    }
)
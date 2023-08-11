from setuptools import setup

setup(
    name="tfmelt",
    version="0.1.0",
    description="TensorFlow Machine Learning Toolbox (TF-MELT)",
    url="https://github.com/nickwimer/tf-melt",
    author="Nicholas T. Wimer",
    author_email="nwimer@nrel.gov",
    license="MIT",
    packages=["tfmelt"],
    install_requires=["tensorflow"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
    ],
)

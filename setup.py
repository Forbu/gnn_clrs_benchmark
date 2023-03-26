"""
Simple setup.py file to install the package
"""
import setuptools

setuptools.setup(
    name="gnn_clrs_reasoning",
    version="0.0.1",
    author="Adrien Bufort",
    author_email="adrienbufort@gmail.com",
    description="Trying to solve clrs reasoning with gnn and reasoning",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=setuptools.find_packages(where="gnn_clrs_reasoning"),
    install_requires=open("requirements.txt", "r").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
    
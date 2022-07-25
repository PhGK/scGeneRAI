from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["torch>=1.0.0"]

setup(
    name="GeneRAI",
    version="0.0.1",
    author="Philipp Keyl",
    author_email="philipp-gerrit.keyl@charite.de",
    description="Tool for the reconstruction of single cell gene regulatory networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/PhGK/scGeneRAI",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

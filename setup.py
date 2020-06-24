import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reval",
    version="0.0.1",
    author="Isotta Landi",
    author_email="isotta.landi@iit.it",
    description="Relative clustering validation to select best number of clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IIT-LAND/reval_clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reval",
    version="1.1.0",
    author="Isotta Landi",
    author_email="isotta.landi@iit.it",
    description="Relative clustering validation to select best number of clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IIT-LAND/reval_clustering",
    download_url="https://github.com/IIT-LAND/reval_clustering/releases/tag/v1.1.2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Documentation": "https://reval.readthedocs.io/en/latest/"},
    install_requires=["numpy",
                      "scipy",
                      "scikit-learn",
                      "umap-learn",
                      "matplotlib"],
    python_requires='>=3.6',
)

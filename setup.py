import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="dbaicdten",
    version="0.0.2",
    auhor="Devyani Kulkarni",
    author_email="devyanikul12345@gmail.com",
    description="High level api for dbaicd10 code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blaze4o4/dbaicdten",
    download_url="https://github.com/blaze4o4/dbaicdten/archive/0.0.2.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
)
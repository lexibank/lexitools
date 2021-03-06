from setuptools import setup, find_packages

# Load requirements, so they are listed in a single place
with open("requirements.txt") as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

setup(
    name="lexitools",
    version="0.1.0.dev0",
    author="Robert Forkel",
    author_email="forkel@shh.mpg.de",
    description="Tools for the lexibank workbench",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="",
    license="Apache 2.0",
    url="https://github.com/lexibank/lexitools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    entry_points={"cldfbench.commands": ["lexitools=lexitools.commands"]},
    platforms="any",
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "dev": ["flake8", "wheel", "twine"],
        "test": ["mock", "pytest>=5", "pytest-mock", "pytest-cov", "coverage>=4.2"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)

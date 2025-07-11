from setuptools import find_packages, setup

setup(
    name="TBD",
    version="0.1.0",
    description="A machine learning project to compare traditional and ML methods for predicting coral bleaching from time series data.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Orlando Timmerman, Robert Rouse, Team Member 3, Team Member 4",
    url="https://github.com/orlando-code/TBD",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
)

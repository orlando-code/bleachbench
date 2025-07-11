from setuptools import find_packages, setup

setup(
    name="bleachbench",
    version="0.1.0",
    description="A machine learning project to compare traditional and ML methods for predicting coral bleaching from time series data.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Orlando Timmerman, Robert Rouse, Akash Verma", "Matt Archer",
    url="https://github.com/orlando-code/bleachbench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
)

from setuptools import setup, find_packages

setup(
    name="testgen",
    version="0.1.0",
    description="A Python package for generating test cases from requirement text.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "langchain-core",
        "langchain-openai",
        "matplotlib",
        "numpy",
        "pandas",
        "openpyxl",
    ],
)

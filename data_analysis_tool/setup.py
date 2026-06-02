from setuptools import setup, find_packages

setup(
    name="data_sanitizer_engine",
    version="0.1.0",
    author="Your Name",
    description="A modular data sanitization, feature engine, and exploration tool",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
)

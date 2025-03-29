from setuptools import setup, find_packages

setup(
    name="fastapi_model_deployment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "pytest",
        "fastapi",
        "uvicorn",
    ],
)

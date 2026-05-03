from setuptools import setup, find_packages

setup(
    name="mlx-code",
    url='https://github.com/JosefAlbers/mlx-code',
    author_email="albersj66@gmail.com",
    author="J Joe",
    license="Apache-2.0",
    version="0.0.2",
    readme="README.md",
    description="Local Coding Agent for Mac",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12.8",
    install_requires=["mlx-lm>=0.31.3", "numpy", "httpx", "pydantic"],
    packages=find_packages(),
    entry_points={"console_scripts": ["mc=mlx_code.main:main", "md=mlx_code.log:main"]},
)

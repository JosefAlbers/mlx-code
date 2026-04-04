from setuptools import setup

setup(
    name="mlx-code",
    url='https://github.com/JosefAlbers/mlx-code',
    author_email="albersj66@gmail.com",
    author="J Joe",
    license="Apache-2.0",
    version="0.0.2a5",
    readme="README.md",
    description="Local Claude Code for Mac",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=["mlx-lm>=0.19.0", "PyYAML"],
    py_modules=["main", "pie"],
    entry_points={"console_scripts": ["mlx-code=main:main"]},
)

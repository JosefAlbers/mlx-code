from setuptools import setup, find_packages

setup(
    name="mlx-code",
    url="https://josefalbers.github.io/mlx-code/",
    project_urls={
        "Source": "https://github.com/JosefAlbers/mlx-code",
        "Issues": "https://github.com/JosefAlbers/mlx-code/issues",
        "Documentation": "https://josefalbers.github.io/mlx-code/",
    },
    author_email="albersj66@gmail.com",
    author="J Joe",
    license="Apache-2.0",
    version="0.0.30",
    readme="README.md",
    description="Coding Agent for Mac",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12.8",
    install_requires=[
        "mlx-lm>=0.31.3; platform_system=='Darwin'",
        "httpx",
        "pydantic",

        "textual>=8.2.7",
        "rich>=15.0.0",
        "starlette",
        "uvicorn",
    ],
    extras_require={"all": [
        "python-lsp-server[all]",
        "GitPython", 
        "pygments",
    ]},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mlc=mlx_code.main:main",
            "mlc-run=mlx_code.repl:main",
            "mlc-log=mlx_code.view_log:main",
            "mlc-git=mlx_code.view_git:main",
        ]
    },
)

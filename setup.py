from setuptools import find_packages, setup

setup(
    name="console-nn",
    version="0.1.0",
    description="Length-unlimited console-log analysis with a tiny FCN",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    license="Apache-2.0",
    license_files=("LICENSE",),
    packages=find_packages(),
    include_package_data=True,
    package_data={"console_nn": ["pretrained/*/*"]},
    entry_points={"console_scripts": ["console-nn=console_nn.cli:main"]},
    install_requires=[],
)

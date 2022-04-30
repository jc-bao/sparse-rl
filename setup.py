from setuptools import setup, find_packages

setup(
    name="sparse_rl",
    version="0.0",
    author="Chaoyi Pan",
    author_email="pcy19@mails.tsinghua.edu.cn",
    packages=find_packages(
        exclude=["experiment"]
    ),
)
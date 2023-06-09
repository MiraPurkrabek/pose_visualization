import os

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCmd(develop):
    def run(self):
        develop.run(self)


class PostInstallCmd(install):
    def run(self):
        install.run(self)


setup(
    name="pose_visualization",
    version="1.0",
    packages=find_packages(exclude=None),
    cmdclass={"develop": PostDevelopCmd, "install": PostInstallCmd},
)

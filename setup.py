from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []

    with open(path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="B0scos",
    author_email="lol.lol.lol.matheus@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
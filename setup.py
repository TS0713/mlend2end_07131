from setuptools import find_packages, setup 
from typing import List


hyphen_e_dot = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open("requirements.txt","r") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements


setup(
    name="mlend2end_07131",
    version="0.0.1",
    author="tsp0713",
    author_email="tsp0713@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),
    
)

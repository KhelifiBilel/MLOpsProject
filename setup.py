from setuptools import find_packages,setup
from typing import List


def get_requirements(path:str)->List[str]:
    requirements=[]
    E_dot='-e .'  # trigger setup.py script
    
    with open(path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if E_dot in requirements:
            requirements.remove(E_dot)

    return requirements
    
    
setup(   
    name='MLOPS Project',
    version='0.0.1',
    author='Bilel',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
) # build the entire project
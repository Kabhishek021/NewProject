from setuptools import find_packages, setup
from typing import List

REQUIRMENTS_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."  # with the help " -e ." this particular command  we can install our setup.py in the requirements.txt

def get_requriments()->List[str]:
    with open(REQUIRMENTS_FILE_NAME) as requriment_file:
        requriment_list = requriment_file.readlines()
    requriment_list = [requriment_name.replace("\n", "") for requriment_name in requriment_list]

    if HYPHEN_E_DOT in requriment_list:
        requriment_list.remove(HYPHEN_E_DOT)

    return requriment_list


setup(name = "MLProjects",
      version = "0.0.1",
      descriptions = "This is machine learning project",
      author = "kumar",
      authod_email = "dummyproject@gmail.com",
      packages = find_packages(),
      install_requires =get_requriments() ,
          )
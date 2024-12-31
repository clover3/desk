import os
from os.path import dirname

data_path = ""
toxicity_project_path = dirname((os.path.abspath(__file__)))
src_root = dirname(toxicity_project_path)
project_root = os.path.abspath(dirname(src_root))
output_root_path = os.path.join(project_root, "outputs")

data_root_path = os.path.join(project_root, "data")



import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "vehicle-tracking-n-detection"

list_of_files = [
    "data/raw/.gitkeep",
    "results/outputs/.gitkeep",
    "results/csv/.gitkeep",
    
    "models/.gitkeep",
    "notebooks/trials.ipynb",
    
    "visual_pipeline/__init__.py",
    "visual_pipeline/stage_01.py",
    "visual_pipeline/stage_02.py",
    
    "training_pipeline/__init__.py",
    "training_pipeline/stage_01.py",
    
    "utils/__init__.py",
    "utils/logger.py",
    "utils/config.py",
    "utils/common.py",
    
    "main.py",
    "test.py",
    
    "params.yaml",
    "DVC.yaml",
    ".env.local",
    ".env.example",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
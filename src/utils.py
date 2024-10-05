import os 
import sys

import pickle
from pathlib import Path

from src.logger import logging
from src.exception import CustomException



def save_object(file_path: Path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
        logging.info(f"Successully saved {file_name} in the location {dir_path}")
        
    except Exception as e:
        logging.info(f"Failed -  saving {file_name} in the location {dir_path}")
        logging.info(str(e))
        CustomException(str(e),sys)
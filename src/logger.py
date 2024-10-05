import logging
import os
from datetime import datetime
from src import src_dir

LOG_FILE = f"{datetime.now().strftime('%Y_%b_%d_%HH_%MM_%SS')}.log"

#logs_path = os.path.join(os.getcwd(),"logs")
#print (os.path.dirname(os.path.abspath("logger.py")))
#print (os.path.dirname(os.path.abspath(__file__)))
logs_path = os.path.join(src_dir,"logs")



os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)



if __name__=="__main__":
    logging.info("Started Logging")


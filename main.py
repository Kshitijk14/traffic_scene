import os, argparse, traceback
from pathlib import Path
from utils.logger import setup_logger
from utils.config import CONFIG
from visual_pipeline.stage_01_detect_n_track import run_stage_01


# configurations
LOG_PATH = CONFIG["LOG_DIR"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")


def main():
    logger = setup_logger("main_logger", LOG_FILE)
    
    # Create CLI.
    parser = argparse.ArgumentParser(description="MAIN WORKFLOW")
    # parser.add_argument("--reset", action="", help="")
    args = parser.parse_args()
    
    try:
        logger.info(" ")
        logger.info("////--//--//----STARTING [PIPELINE 01] CV PIPELINE----//--//--////")
        
        try:
            logger.info(" ")
            logger.info("----------STARTING [STAGE 01]----------")
            run_stage_01()
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 01]----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 01]: {e}")
            logger.debug(traceback.format_exc())
            return

        logger.info("////--//--//----FINISHED [PIPELINE 01] CV PIPELINE----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"ERROR RUNNING [PIPELINE 01] CV PIPELINE: {e}")
        logger.debug(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
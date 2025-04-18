import os
import logging
# from llm_wf_llama3_1 import setup_logger

def setup_logger(log_file_path):
    """Set up a logger for a specific log file."""
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.DEBUG)  # Make sure it's set to DEBUG or INFO level
    
    # Avoid adding multiple handlers if already added
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)  # Ensure it captures DEBUG and lower levels
        
        # Formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)
    
    return logger

def test_main():
    model = "llama3.1"

    log_dir = f"Error_Analysis/{model}"
    os.makedirs(log_dir, exist_ok=True)

    # Setup logger
    logging_name = f"{log_dir}/logging/test_log.log"
    logger = setup_logger(logging_name)

    # Log some initial test messages
    logger.info("Starting the process")
    logger.debug("Debugging logger")

    # Your workflow code starts here
    logger.info("Starting project creation")

    # Example of logging during the process
    logger.info("Processing some data here...")

    # Continue with the rest of the script logic
    # Explicitly flush handlers to ensure logs are written to file
    for handler in logger.handlers:
        handler.flush()
    
    print("Logging complete. Check the log file for messages.")

if __name__ == "__main__":
    test_main()

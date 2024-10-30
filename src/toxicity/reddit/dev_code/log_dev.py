import logging


def setup_file_logger(logger_name, log_file):
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Prevent propagation to root logger
    logger.propagate = False

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    # Set up logger
    logger = setup_file_logger('my_app', 'app.log')

    # Test logging
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')
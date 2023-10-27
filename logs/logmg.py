import logging, sys, os

class LogManager:
    def __init__(self, src):
        self.src = src
        self.logger = self.setup_logging()

    def setup_logging(self):
        log_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_file = f'{save_folder}/{self.src}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        # console_handler = logging.StreamHandler(sys.stderr)
        # console_handler.setFormatter(log_formatter)
        logger = logging.getLogger(self.src)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)

        return logger
    
    def log(self, message, *args):
        self.logger.info(message, *args)

# Create separate LogManager instances for different sources
save_folder = "logs"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

m = LogManager("monitoring")
i = LogManager("infering")


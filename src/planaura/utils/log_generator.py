import logging
import sys


def setup_file_logging(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfile_path)
    ch = logging.StreamHandler(sys.stdout)

    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    class PrintLogger(object):
        def write(self, msg):
            if msg != '\n':
                logging.info(msg.strip())

        def flush(self):
            pass

    sys.stdout = PrintLogger()
    sys.stderr = PrintLogger()

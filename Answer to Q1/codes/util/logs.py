import logging
import colorlog

'''
Format example:
 [INFO] test() at /path/to/file:6
    Log message at next line
'''

fmt_str = ' [%(levelname)s] %(funcName)s() at %(pathname)s:%(lineno)s\n\t%(message)s'

console_formatter = colorlog.ColoredFormatter(
    '%(log_color)s' + fmt_str + '%(reset)s')
file_formatter = logging.Formatter(fmt_str)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler('run_L2.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)  # Enable logging to console
logger.addHandler(file_handler)  # Enable logging to file


if __name__ == '__main__':
    logger.warning("Test msg")
    logger.info("Test msg")
    logger.debug("Test msg")

import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, formatString, dateFormat=None, useColor=False):
        logging.Formatter.__init__(self, formatString, datefmt=dateFormat)
        self.useColor = useColor

    def format(self, record):
        levelname = record.levelname
        msg = record.msg
        if type(msg) == int or type(msg) == float: #转变成为str
            msg = str(msg)
            record.msg = msg
        if self.useColor and levelname in COLORS:
            formated = logging.Formatter.format(self, record)
            return COLOR_SEQ % (30 + COLORS[levelname]) + formated + RESET_SEQ
        else:
            return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    COLOR_FORMAT = "%(asctime)s.%(msecs)d %(filename)s_%(lineno)d(%(levelname)s): %(message)s"
    FILE_COLOR_FORMAT = "%(asctime)s %(filename)s_%(lineno)d(%(levelname)s): %(message)s"

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)
        color_formatter = ColoredFormatter(self.COLOR_FORMAT, dateFormat="%H:%M:%S", useColor=True)
        file_color_formatter = ColoredFormatter(self.FILE_COLOR_FORMAT, useColor=False)
        # log to console
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        console.setLevel("INFO")
        self.addHandler(console)
        # log to file
        file = logging.FileHandler("logging.log",encoding='utf-8')
        file.setFormatter(file_color_formatter)
        file.setLevel("DEBUG")
        self.addHandler(file)
        return

logging.setLoggerClass(ColoredLogger)
FileLogger = logging.getLogger(__name__)
FileLogger.setLevel(logging.DEBUG)

import logging
from termcolor import colored
from datetime import datetime
import sys

# set up logger
class _MyFormatter(logging.Formatter):

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

logger = logging.getLogger('SXC')
logger.propagate = False
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
logger.addHandler(handler)


# set up log_file
_file = None
_run_name = None
_slack_url = None
_format = '%Y-%m-%d %H:%M:%S.%f'

def init(filename,run_name,slack_url=None):
    global _file, _run_name, _slack_url
    _close_logfile()
    _file = open(filename, 'a', encoding="utf-8")
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting new training run\n')
    _file.write('-----------------------------------------------------------------\n')
    _run_name = run_name
    _slack_url = slack_url

def _close_logfile():
  global _file
  if _file is not None:
    _file.close()
    _file = None

def log(msg):
  logger.info(msg)
  if _file is not None:
    _file.write('[%s]  %s\n' % (datetime.now().strftime(_format)[:-3], msg))
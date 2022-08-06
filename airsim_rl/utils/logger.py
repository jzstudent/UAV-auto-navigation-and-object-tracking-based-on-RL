# -----------------------------------------------------------------------------
#   @brief:
#       The logger here will be called all across the project. It is inspired
#   by Yuxin Wu (ppwwyyxx@gmail.com)
#
#   @author:
#       Tingwu Wang, 2017, Feb, 20th
# -----------------------------------------------------------------------------

import logging
import sys
import os
import datetime
from termcolor import colored

__all__ = ['set_file_handler']  # the actual worker is the '_logger'


class _MyFormatter(logging.Formatter):
    '''
        @brief:
            a class to make sure the format could be used
    '''

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'

        if record.levelno == logging.WARNING:
            fmt = date + ' ' + \
                colored('WRN', 'red', attrs=[]) + ' ' + msg
        elif record.levelno == logging.ERROR or \
                record.levelno == logging.CRITICAL:
            fmt = date + ' ' + \
                colored('ERR', 'red', attrs=['underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg

        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt

        return super(self.__class__, self).format(record)


_logger = logging.getLogger('joint_embedding')
_logger.propagate = False
_logger.setLevel(logging.INFO)

# set the console output handler
con_handler = logging.StreamHandler(sys.stdout)
con_handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
_logger.addHandler(con_handler)


# set the file output handler
def set_file_handler(path=None, prefix='', time_str=''):
    if time_str == '':
        file_name = prefix + \
            datetime.datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p") + '.log'
    else:
        file_name = prefix + time_str + '.log'

    if path is None:
        mod = sys.modules['__main__']
        path = os.path.join(os.path.abspath(mod.__file__), '..', '..', 'log')
    else:
        path = os.path.join(path, 'log')
    path = os.path.abspath(path)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, file_name)

    file_handler = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    file_handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(file_handler)

    _logger.info('Log file set to {}'.format(path))
    return


_LOGGING_METHOD = ['info', 'warning', 'error', 'critical',
                   'warn', 'exception', 'debug']

# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)

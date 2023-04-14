# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/14 9:38
# @File: WriteLog.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import sys
import os
import sys
import datetime


def make_print_to_file(path='./'):
    """
    path: it is a path for save your log about function print
    example:
    use make_print_to_file() and the all the information of function print , will be write in to a log file
    :return:
    """
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path = os.path.join(path, filename)
            self.log = open(self.path, "a", encoding='utf8')
            print("save:", os.path.join(self.path, filename))

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    print(fileName.center(60, '*'))


if __name__ == '__main__':
    make_print_to_file(path='./')


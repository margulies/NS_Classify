#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys


class Logger():

    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'w')
        self.old = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)
        self.log.flush()

    def end(self):
        self.log.close()
        sys.stdout = self.old

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def show(self, text):
        self.stdout.write(text)
        self.stdout.flush()


class ProgressBar():

    def __init__(self, total):
        self.total = total
        self.current = 0.0

    def update_progress(self, progress):
        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        if sys.stdout.__class__.__name__ == 'Logging':
            sys.stdout.show(display)
        else:
            sys.stdout.write(display)
            sys.stdout.flush()

    def next(self):
        self.update_progress(int((self.current) / self.total * 100))

        if self.current == self.total:
            self.reset()
        else:
            self.current = self.current + 1

    def reset(self):
        print ""
        self.current = 0.0


def ix_subset(dataset, subset):
    return [i for i, x in enumerate(dataset.get_feature_names()) if x in subset]

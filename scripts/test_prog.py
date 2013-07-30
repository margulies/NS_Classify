import time
import itertools
import sys

def go():

	items = range(0,10)

	prog = 0.0
	total = len(items)

	update_progress(0)

	for i in items:
		time.sleep(0.2)
		prog = prog + 1
		update_progress(int(prog / total * 100))

def update_progress(progress):
	sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress / 10), progress))
	sys.stdout.flush()

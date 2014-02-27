
from multiprocessing import Process, Queue
import time

def do_something(r, queue):
	time.sleep(10)

	queue.put(r*r)
	return True

results = []
queue = Queue()
for i in xrange(1, 20):
	res = Process(target = do_something, args=(i, queue))
	res.start()
	results.append(res)

for r in results:
	r.join()

while not queue.empty():
	print queue.get()


import threading
import time
import logging
import random

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )


class Counter(object):
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.value = start

    def increment(self):
        logging.debug('Waiting for a lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired a lock')
            self.value = self.value + 1
        finally:
            logging.debug('Released a lock')
            self.lock.release()


def face_recognition(c):
    for i in range(2):
        r = random.random()
        logging.debug('Sleeping %0.02f', r)
        time.sleep(r)
        c.increment()
    logging.debug('Done')


def voice_recognition(c):
    for i in range(2):
        r = random.random()
        logging.debug('Sleeping %0.02f', r)
        time.sleep(r)
        c.increment()
    logging.debug('Done')

if __name__ == '__main__':
    counter = Counter()
    # for i in range(2):
    t1 = threading.Thread(target=face_recognition, args=(counter,))
    t2 = threading.Thread(target=face_recognition, args=(counter,))
    t1.start()
    t2.start()

    logging.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    # for t in threading.enumerate():
    #     if t is not main_thread:
    #         t.join()
    # logging.debug('Counter: %d', counter.value)




# import logging
# import threading
# import time
#
#
# lock = threading.Lock()
#
# def get_first_part():
#     lock.acquire()
#     try:
#         print('first part')
#     finally:
#         lock.release()
#     return '1'
#
# def get_second_part():
#     lock.acquire()
#     try:
#         print('second part')
#     finally:
#         lock.release()
#     return '2'
#
#
# def get_both_parts():
#     first = get_first_part()
#     second = get_second_part()
#     return first, second
#
# f,s = get_both_parts()
# print(f,s)
#
# def face_function(name):
#     logging.info("Face%s: starting", name)
#     time.sleep(102)
#     logging.info("face %s: finishing", name)
#
#
#
# def voice_function(name):
#     logging.info("voice %s: starting", name)
#     time.sleep(2)
#     logging.info("voice %s: finishing", name)
#
#
#
# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#
#     logging.basicConfig(format=format, level=logging.INFO,
#
#                         datefmt="%H:%M:%S")
#
#     logging.info("Main    : before creating thread")
#
#     x = threading.Thread(target=face_function, args=(1,))
#     y = threading.Thread(target=voice_function, args=(2,))
#     logging.info("Main    : before running thread")
#
#     x.start()
#     y.start()
#
#     logging.info("Main    : wait for the thread to finish")
#
#     # x.join()
#
#     logging.info("Main    : all done")

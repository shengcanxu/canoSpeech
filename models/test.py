# /usr/bin/python3
# coding:utf8

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import os, time, random


def task(n, p):
    print('%s:%s is running' % (threading.currentThread().getName(), os.getpid()))
    time.sleep(2 * n)

    return n ** 2


def parse_page(res):  # 此处的res是一个p.submit获得的一个future对象，不是结果
    res = res.result()  # res.result()拿到的才是对应的结果
    print('result : %d' % (res))
    time.sleep(2)


def main():
    p = ProcessPoolExecutor()  # 不填则默认为cpu的个数*5
    l = []
    start = time.time()
    for i in range(10):
        obj = p.submit(task, (i, p)).add_done_callback(parse_page)
        l.append(obj)
    p.shutdown()
    print('=' * 30)
    print(time.time() - start)


def main_as_complete():
    p = ProcessPoolExecutor()  # 不填则默认为cpu的个数*5
    l = []
    start = time.time()
    for i in range(10):
        obj = p.submit(task, i)
        l.append(obj)

    print("===============")
    print(len(l))
    for future in as_completed(l):
        print("++++++++++")
        print(future.result())

    print('=' * 30)
    print(time.time() - start)


if __name__ == '__main__':
    # main()
    main_as_complete()


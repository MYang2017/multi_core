# multi_core
# template for parallel scripting
# another example with buffered array in memory:
# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

# ------------------------------------------------ code begins ---------------

from functools import partial
import time
import multiprocessing as mp



def base_for(index):
    time.sleep(0.001)


def stupid_for():
    for i in range(10000):
        base_for(i)


def smart_mult():
    pool = mp.Pool(processes=mp.cpu_count() - 2)
    func = partial(base_for)
    pool.map(func, range(10000))
    pool.close()
    pool.join()


def main():
    t0 = time.time()
    stupid_for()
    t1 = time.time()
    t_stupid = t1-t0

    t0 = time.time()
    smart_mult()
    t1 = time.time()
    t_smart = t1 - t0

    print('Processing time slow: {}'.format(t_stupid))
    print('Processing time multi: {}'.format(t_smart))


if __name__ == '__main__':
    main()

import numpy as np
import multiprocessing as mp
from functools import partial
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, RawArray


# multiprocessing template to accelerate for loop operation
# X is the common big data to share for sub-processes
# Y is the output array used by sub-processes

# A global dictionary storing the variables passed from the initializer function.
var_dict = {}


def init_worker(X, X_shape, Y, Y_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    var_dict['Y'] = Y
    var_dict['Y_shape'] = Y_shape


def worker_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    Y_np = np.frombuffer(var_dict['Y']).reshape(var_dict['Y_shape'])

    # get index
    xi = i % var_dict['X_shape'][2]
    yi = (i - xi)//var_dict['X_shape'][2]

    dataPix = X_np[:, yi, xi]
    low = [0.1, 0.1, 0.1, 0.001]
    up = [255, np.pi, 20, 0.03]
    params, params_covariance = optimize.curve_fit(test_cos2_func, np.arange(256), dataPix, bounds=(low, up))

    Y_np[yi, xi, :] = params
    np.copyto(Y_np, Y_np)
    return


def perPixelFit_multiCore(loadname, savename):
    # We need this check for Windows to prevent infinitely spawning new child
    # processes.

    Config, path = setup()
    images = np.load(Config['path'] + '/' + loadname + '.npy')
    levels = 256
    X_shape = (images.shape[0], images.shape[1], images.shape[2])
    Y_shape = (images.shape[1], images.shape[2], 4)
    results = np.zeros(Y_shape)
    # Randomly generate some data
    data = images
    X = RawArray('d', X_shape[0] * X_shape[1] * X_shape[2])
    Y = RawArray('d', Y_shape[0] * Y_shape[1] * Y_shape[2])
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    Y_np = np.frombuffer(Y).reshape(Y_shape)
    # Copy data to our shared array.
    np.copyto(X_np, data)
    np.copyto(Y_np, results)
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=mp.cpu_count() - 8, initializer=init_worker, initargs=(X, X_shape, Y, Y_shape)) as pool:
        pool.map(worker_func, range(X_shape[1]*X_shape[2]))

    res = np.frombuffer(Y).reshape(Y_shape)
    print(res)
    np.save(Config['path'] + '/' + savename + '.npy', res)


def main():
    median_filter_width = 300
    perPixelFit_multiCore('images_median_filter_' + str(median_filter_width), 'slm_cos_fit_data_' + str(median_filter_width))


if __name__ == '__main__':
    main()




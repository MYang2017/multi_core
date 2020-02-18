import numpy as np
import multiprocessing as mp
from functools import partial
from HSD.spatial_modem2 import checkerboard, estimate_translation, import_image
from optsimpy.spatial_modem import interp_grid, interp2D
from HSD.image.measure import symbol_clock
from optsimpy.utils.import_config import import_config_using_arguments
import time
from socket import gethostname
from HSD.session import setup
import json
from HSD.Camera_calibration.camera import load_transform
import HSD.image.filter as filt
import HSD.image.operations as image
from HSD.modulate import create_xygrid, creategrid

import HSD.expt as expt
import HSD.image.operations as imageops
import matplotlib.pyplot as plt
from scipy import optimize
from skimage.filters.rank import median as rank
from skimage.morphology import disk
import imageio
from multiprocessing import Pool, RawArray


# A global dictionary storing the variables passed from the initializer function.
var_dict = {}


def process_image(pagenumber, Config, median_filter_width):
    iteration = 0
    print('Importing image: ', pagenumber, flush=True)
    path = Config['path']
    load_path = '{}/page_{}'.format(path, pagenumber)
    img = imageio.imread('{}/cam_{}.png'.format(load_path, iteration))
    img = np.fliplr(img)
    log_file = '{}/../calibration/crop.json'.format(path)
    with open(log_file, "r") as read_file:
        crop = json.load(read_file)
    cropped_img = imageops.crop(img, crop['width'], crop['height'], crop['x0'], crop['y0'],
                                method=crop['method'])

    with open(load_path + '/modulate_encode.json', "r") as read_file:
        encode_params = json.load(read_file)

    # ref_xy = create_xygrid(Config['Sx'],
    #                        Config['Sy'],
    #                        1,
    #                        1,
    #                        0,  # encode_params['x_offset']*encode_params['sps_x'],
    #                        0  # encode_params['y_offset']*encode_params['sps_y']
    #                        )

    af = load_transform('af.npy', path + '/../calibration')
    tf = load_transform('tf.npy', path + '/../calibration')

    # image.plot(img, 'Camera image', colorbar=True, run=plot)

    # ## Apply low pass filter to reduce noise
    # img = filt.gaussian_filter_fft(cropped_img, sigma=Config['data_filter'], pad=True)
    img = cropped_img

    ## generate pixel coordinate
    ref_xy = np.load('{}/../calibration/xy.npy'.format(path))
    xy = tf(af(ref_xy))
    x_1pix = np.array(np.linspace(np.min(xy[:,0]), np.max(xy[:,0]), Config['Sx']), dtype='int')
    y_1pix = np.array(np.linspace(np.min(xy[:,1]), np.max(xy[:,1]), Config['Sy']), dtype='int')
    xv, yv = np.meshgrid(x_1pix, y_1pix)

    img_s = img[yv, xv]
    plt.figure('distorted')
    plt.imshow(img_s)

    img_s = rank(img_s, disk(median_filter_width))

    # plt.figure('filtered')
    # plt.imshow(img_s)
    # plt.show()

    return img_s


def generate_images_dot_npy(savename, median_filter_width):
    print("Number of processors: ", mp.cpu_count())
    print("Host: ", gethostname())

    t0 = time.time()
    Config, path = setup()

    pool = mp.Pool(processes=mp.cpu_count() - 1)
    func = partial(process_image, Config=Config, median_filter_width=median_filter_width)
    results = pool.map(func, range(Config['pages']))
    pool.close()
    pool.join()

    out = np.zeros((Config['pages'], Config['Sy'], Config['Sx']))
    for index in range(Config['pages']):
        out[index, :, :] = results[index]

    filename = Config['path'] + '/images_' + savename + '.npy'
    np.save(filename, out)
    print('Processing time: ', time.time() - t0, flush=True)


def check_image_dot_npy(fileName, no):
    Config, path = setup()
    images = np.load(Config['path'] + '/' + fileName + '.npy')

    print(images.dtype)

    plt.figure()
    plt.imshow(images[no,:,:])
    plt.show()


def plot_avg_vs_level(savename):
    Config, path = setup()
    levels = 256 # images.shape[0]
    images = np.load(Config['path'] + '/' + savename + '.npy')

    avgs = np.zeros(levels)
    for avg in np.arange(levels):
        avgs[avg] = np.average(images[avg,:,:])

    # save data
    filename = Config['path'] + '/avg_vs_level.npy'
    np.save(filename, avgs)

    # fit
    avgs = np.load(Config['path'] + '/avg_vs_level.npy')
    params, params_covariance = optimize.curve_fit(test_cos2_func, np.arange(levels), avgs, p0=[80, 2, 15, 0.01])
    print(params)
    filename = Config['path'] + '/fit_data.npy'
    np.save(filename, params)

    plt.figure()
    plt.scatter(np.arange(levels), avgs)
    plt.plot(np.arange(levels), test_cos2_func(np.arange(levels), params[0], params[1], params[2], params[3]))
    plt.show()


def plot_hist(savename):
    Config, path = setup()
    images = np.load(Config['path'] + '/images_' + savename + '.npy')
    image = images[10, :, :]
    plt.hist((np.squeeze(image.astype())), 10)
    plt.show()


def plot_heatmap(no):
    Config, path = setup()
    images = np.load(Config['path'] + '/images_no_filter.npy')
    avgs = np.load(Config['path'] + '/avg_vs_level.npy')
    image = images[no, :, :]
    heat = np.abs(image - avgs[no])

    plt.clf()
    plt.imshow(heat)
    plt.show()


def reduce_images_resolution(symbol):
    Config, path = setup()
    images = np.load(Config['path'] + '/images.npy')

    filename = Config['path'] + '/images_' + str(symbol) + '.npy'
    np.save(filename, images)
    return


def test_cos2_func(x, a, b, c, w):
    return a * np.cos(w*x-b)**2 + c


def check_params():
    if True:
        Config, path = setup()
        # perpix = np.load(Config['path'] + '/slm_fit_data_pixel.npy')
        perpix = np.load(Config['path'] + '/test_multi_fit.npy')
        for it in range(4):
            plt.figure()
            y = perpix[:, :, it].flatten()
            plt.scatter(range(len(y)), y)
        plt.show()


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


def process_with_different_filter_width():
    for width in [200, 250, 450, 500]:
        # # generate 256-ny-nx npy
        median_filter_width = width
        generate_images_dot_npy('median_filter_' + str(median_filter_width), median_filter_width)

        # process distorted and filtered images
        perPixelFit_multiCore('images_median_filter_' + str(median_filter_width), 'slm_cos_fit_data_' + str(median_filter_width))


def check_amp_filter_width():
    Config, path = setup()
    widths = [200, 250, 300, 350, 450, 500]
    maxs = np.zeros(len(widths))
    mins = maxs
    for idx, width in enumerate(widths):
        params = np.load(Config['path'] + '/' + 'images_median_filter_' + str(width) + '.npy')
        maxs[idx] = np.max(params[:, :, 0])
        mins[idx] = np.min(params[:, :, 0])

    plt.figure()
    plt.plot(maxs)
    plt.plot(mins)
    plt.show()


def main():
    # some utility plot functions
    # check_image_dot_npy('images_smoothed', 10)
    # plot_avg_vs_level('images_meidan_filter')
    # plot_hist('no_filtr')
    # plot_heatmap(10)
    # check cos fit parameters
    # check_params()

    # process_with_different_filter_width()
    check_amp_filter_width()


if __name__ == '__main__':
    main()




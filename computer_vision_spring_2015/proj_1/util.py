
# Common functions used for the programming assignment

import os
import ntpath
import math
import numpy
import Image
import scipy.ndimage.filters

inf = float('inf')
PI = math.pi

round_vectorized = numpy.vectorize(round)

def system(cmd):
    pass
    print cmd
    os.system(cmd)

def save_image(image, name): 
    final_output = clamp_array(image) 
    final_output = final_output.astype('uint8') 
    Image.fromarray(final_output).save(name) 

def list_from_set(s):
    return [ e for e in s ]

def mean(l):
    return sum(l) / float(len(l))

def list_dir_abs(basepath):
    return map(lambda x: os.path.join(basepath, x), os.listdir(basepath))

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_containing_folder(path):
    head, tail = ntpath.split(path)
    return head

def makedirs_recursive(dirname):
    if not os.path.exists(dirname):
        makedirs_recursive(get_containing_folder(dirname))
        os.makedirs(dirname)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def G(x, sigma):
    return math.exp(-(x*x)/(2*(sigma*sigma))) / sqrt(2*PI*(sigma*sigma));

def G2(x, y,sigma):
    return math.exp(-((y*y)+(x*x))/(2*(sigma*sigma))) / (2*PI*(sigma*sigma));

def divide(top, bottom):
    if bottom == 0.0:
        return inf
    return top / bottom

def multiply(a, b):
    return a*b

def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def radians_to_degrees(radian_value):
    return radian_value*180/PI

def downsample_2d(I0, downsample_factor=2):
    I = numpy.array(I0,dtype='float')
    return I[::2,::2]

def convert_to_grayscale(I0):
    I = numpy.array(I0,dtype='float')
    I[:,:,0] *= 0.299
    I[:,:,1] *= 0.5870
    I[:,:,2] *= 0.1140
    I_grayscale = numpy.mean(I, axis=2)
    return I_grayscale

def convolve(I0, k, zero_borders=False): # Convolves RGB image with 2D kernel 
    assert len(k.shape) == 2, "Kernel must be 2D"
    k_h, k_w = k.shape
    assert k_h % 2 == 1, "Kernel must have odd height"
    assert k_w % 2 == 1, "Kernel must have odd width"
    
    I = numpy.zeros(I0.shape)
    if len(I0.shape)==2:
        I_h, I_w = I.shape
        if zero_borders:
            I = scipy.ndimage.filters.convolve(I0, k, mode='constant', cval=0.0)
        else:
            I = scipy.ndimage.filters.convolve(I0, k, mode='nearest')
    else:
        I_h, I_w, I_channels = I0.shape
        for channel in xrange(I_channels):
            if zero_borders:
                I[:,:,channel] = scipy.ndimage.filters.convolve(I0[:,:,channel], k, mode='constant', cval=0.0)
            else:
                I[:,:,channel] = scipy.ndimage.filters.convolve(I0[:,:,channel], k, mode='nearest')
    
    return I

def normalize(array0):
    assert len(array0.shape) in [2,3], "normalize() is only supported for 2D and 3D arrays"
    array = None
    if len(array0.shape) == 2:
        array_sum = numpy.sum(array0)
        array = numpy.array(array0)
        h, w = array0.shape
        for y in xrange(h):
            for x in xrange(w):
                array[y,x] /= array_sum
    elif len(array0.shape) == 3:
        array = numpy.array(array0)
        h, w, channels = array0.shape
        for z in xrange(channels):
            array_sum = numpy.sum(array0[:,:,z])
            for y in xrange(h):
                for x in xrange(w):
                    array[y,x,z] /= array_sum
    assert array is not None
    return array

def clamp(x, min_val=0, max_val=255):
    return max( min(x, max_val), min_val)

clamp_array = numpy.vectorize(clamp)

def get_gaussian_kernel(dim, sigma):
    assert dim % 2==1, "Gaussian kernel must be of odd dimension"
    
    kernel = numpy.zeros(shape=(dim,dim))
    
    for y in xrange(dim):
        for x in xrange(y+1):
            weight = G2(y-dim/2, x-dim/2, sigma)
            kernel[y,x] = weight
            if x != y:
                kernel[x,y] = kernel[y,x]
    kernel = normalize(kernel)
    return kernel


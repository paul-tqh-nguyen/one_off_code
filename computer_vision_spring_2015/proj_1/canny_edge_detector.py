#!/usr/bin/python

# Canny Edge Detector

# Standard Libraries
import sys
import os
import pdb
import time
import math
import numpy
import Image
import scipy.ndimage.filters

# Non-Standard Libraries
from util import *

KERNEL_DIM = 5 # the denoising kernels are going to be 5x5, we will normalize the kernel to make sure the coefficients sum to 1.0

def usage():
    # Sample Usage: ./canny_edge_detector.py building.jpg 9 35 75
    print >> sys.stderr, 'python '+__file__+' input_image denoising_sigma low_threshold high_threshold out_dir'
    sys.exit(1)

def main():
    if len(sys.argv) < 6:
        usage()
    
    # Get Params
    input_image_location = os.path.abspath(sys.argv[1])
    denoising_sigma = float(sys.argv[2]) # the sigma to use for denoising
    low_threshold = float(sys.argv[3]) # the lower threshold to be used for hysteresis thresholding
    high_threshold = float(sys.argv[4]) # the higher threshold to be used for hysteresis thresholding 
    out_dir = os.path.abspath(sys.argv[5]) # the output directory
    makedirs(out_dir)
    output_image_location = os.path.join(out_dir,'out.png') # the output image
    
    I = numpy.asarray(Image.open(input_image_location)).astype('float') # I'm using int values in the range [0,255], but I want to manipulate them as floats to not lose precision
    I_h, I_w, I_channels = I.shape # get input image dimensions
    
    total_run_time_start = time.time()
    
    # Filtered Gradient ##############################################################################    
    # Denoise with Gaussian Filter
    denoise_start = time.time() # I'm profiling my code to see how fast things run
    denoising_kernel = get_gaussian_kernel(KERNEL_DIM, denoising_sigma) # see util.py on how I generate the gaussian kernel
    I_denoised = convolve(I, denoising_kernel) # see util.py on how the convolution was implemented
    save_image(I,os.path.join(out_dir,'input.png'))
    save_image(I_denoised,os.path.join(out_dir,'denoised.png'))
    denoise_end = time.time()
    
    # Find x and y components of gradient
    find_gradient_start = time.time()
    I_denoised_grayscale = convert_to_grayscale(I_denoised) # we're using teh gray scale images to calculate the sobel filtered image and to find the edges
    sobel_x = numpy.array([[-1, 0, 1], # these are the sobel filter kernels
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype='float')
    sobel_y = numpy.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype='float')
    Fx = convolve(I_denoised_grayscale, sobel_x) # convolving with the sobel filter kernels
    Fy = convolve(I_denoised_grayscale, sobel_y)
    save_image(Fx,os.path.join(out_dir,'Fx.png'))
    save_image(Fy,os.path.join(out_dir,'Fy.png'))
    find_gradient_end = time.time()
    
    # Compute edge strength
    compute_edge_strength_start = time.time()
    F = numpy.sqrt( numpy.square(Fx)+numpy.square(Fy) ) # using vectorized operations to calculate the magnitudes of the edges
    save_image(F,os.path.join(out_dir,'magnitude.png'))
    compute_edge_strength_end = time.time()
    
    # Compute edge orientation
    compute_edge_orientation_start = time.time()
    radians_to_degrees_vectorized = numpy.vectorize(radians_to_degrees) # vectorizing my radians to degrees func (see util.py for implementation) 
    atan_vectorized = numpy.vectorize(math.atan) 
    divide_vectorized = numpy.vectorize(divide)
    D = radians_to_degrees_vectorized(atan_vectorized(divide_vectorized(Fy,Fx))) # using vectorized operations to determine the angle, makes things run much faster 
    compute_edge_orientation_end = time.time()
    
    # Nonmaximum Suppression #########################################################################
    # Compute quantized edge orientation
    nonmaximal_suppression_start = time.time()
    D_star = round_vectorized(D/45.0)*45.0 # I quantize the angles to be multiples of 45 degrees
    
    # Suppression
    angle_to_rel_pixels_dict = { -90 : (( 0,-1),( 0, 1)), # here is a dict I use to find corresponding relative coordinates of angles
                                 -45 : (( 1,-1),(-1, 1)), 
                                   0 : (( 1, 0),(-1, 0)), 
                                  45 : (( 1, 1),(-1,-1)), 
                                  90 : (( 0, 1),( 0,-1)) } # stored as (x,y) pairs
    edge_map = numpy.array(F) # referred to as I in the assignment description
    for y in xrange(I_h): 
        for x in xrange(I_w): 
            pos_dir, neg_dir = angle_to_rel_pixels_dict[D_star[y,x]]
            pos_x = pos_dir[0] + x # getting the coordinates of the positive and negative directions along the orientation direction
            pos_y = pos_dir[1] + y
            neg_x = neg_dir[0] + x
            neg_y = neg_dir[1] + y
            
            # the actual nonmaximum supression takes place here
            if pos_x > 0 and pos_x < I_w and pos_y > 0 and pos_y < I_h:
                if F[y,x] < F[pos_y,pos_x]: # make sure it's bigger than the pixel in the positive direction
                    edge_map[y,x] = 0
            if neg_x > 0 and neg_x < I_w and neg_y > 0 and neg_y < I_h:
                if F[y,x] < F[neg_y,neg_x]: # make sure it's bigger than the pixel in the negative direction
                    edge_map[y,x] = 0
    save_image(edge_map,os.path.join(out_dir,'suppressed_magnitude.png'))
    nonmaximal_suppression_end = time.time()
    
    # Hysteresis Thresholding #########################################################################
    hysteresis_thresholding_start = time.time()
    all_rel_directions = reduce(lambda a,b:a+b, map(lambda e: list(e), [f for f in set(angle_to_rel_pixels_dict.values())])) # this is just a list of all the relative positions of the 8 pixels surrounding any one pixel
    visited_pixels = (edge_map > high_threshold) # a buffer of values that are above our high threshold, this will be our starting point for our final image. We will add on all surrounding pixels that form a chain with these above-the-high-threshold pixels in the code below
    pixels_to_explore = [] # our todo list of pixels
    final_output = numpy.zeros(I.shape, dtype='float') # our final output buffer is initially all zeros
    y_values, x_values = map(list, visited_pixels.nonzero()) # grab all the y and x values of the pixels above the threshold.
    for (y,x) in zip(y_values, x_values): 
        final_output[y,x] = edge_map[y,x] # we add the pixels above the threshold to our final ans
        for rel_x, rel_y in all_rel_directions: # for all the surrounding pixels
            to_explore_x = x+rel_x
            to_explore_y = y+rel_y
            if to_explore_x > 0 and to_explore_x < I_w and to_explore_y > 0 and to_explore_y < I_h and not visited_pixels[to_explore_y,to_explore_x]: # if the pixel is in bounds and we haven't already added this pixel to our final answer, we should check it later
                pixels_to_explore.append( (to_explore_x,to_explore_y) )  
    
    count = 0
    while len(pixels_to_explore) > 0: # while have more pixels to check
        count += 1
        x, y = pixels_to_explore.pop()
        visited_pixels[y,x] = True # make this pixel as checked so we don't check it again
        if edge_map[y,x] > low_threshold: # this pixel is from our todo list, which means it neighbors a chain with at least one pixel above the high threshold. Thus, if this neighboring pixel is above the low threshold, we consider it as part of the chain
            final_output[y,x] = edge_map[y,x] # add this pixel to our final ans
            for rel_x, rel_y in all_rel_directions: # add it's neighbors to our todo list
                to_explore_x = x+rel_x
                to_explore_y = y+rel_y
                if to_explore_x > 0 and to_explore_x < I_w and to_explore_y > 0 and to_explore_y < I_h and not visited_pixels[to_explore_y,to_explore_x]: # if any neighboring pixels are in bound and haven't been checked yet, they are potential candidates for our final ans.
                    pixels_to_explore.append( (to_explore_x,to_explore_y) )
    hysteresis_thresholding_end = time.time()
    
    total_run_time_end = time.time()
    
    # Save output image
    final_output = clamp_array(round_vectorized(final_output)) # round our values and clamp values to be within range
    final_output = final_output.astype('uint8') # convert our values to be the correct type. 
    Image.fromarray(final_output).save(output_image_location) # save the image
    
    # print performance information
    print 
    print "Canny Edge Detector Complete!"
    print "Denoising Time: "+str(denoise_end-denoise_start)+" seconds"
    print "Gradient Calculation Time: "+str(find_gradient_end-find_gradient_start)+" seconds"
    print "Edge Strength Calculation Time: "+str(compute_edge_strength_end-compute_edge_strength_start)+" seconds"
    print "Edge Orientation Calculation Time: "+str(compute_edge_orientation_end-compute_edge_orientation_start)+" seconds"
    print "Nonmaximal Suppression Time: "+str(nonmaximal_suppression_end-nonmaximal_suppression_start)+" seconds"
    print "Hysteresis Thresholding Calculation Time: "+str(hysteresis_thresholding_end-hysteresis_thresholding_start)+" seconds"
    print "Total Run Time: "+str(total_run_time_end-total_run_time_start)+" seconds"
    print 

if __name__ == '__main__':
    main()


#!/usr/bin/python

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

KERNEL_DIM = 5
BOX_THICKNESS = 2
BOX_COLOR = [0,0,255]

def get_smaller_eigenvalue(F_x_squared_val, F_y_squared_val, F_xy_val):
    eigen_values, eigen_vectors = numpy.linalg.eig(numpy.array(
                                                                [[F_x_squared_val,F_xy_val],
                                                                 [F_xy_val,F_y_squared_val]] 
                                                              ))
    return min(eigen_values)

def usage():
    # Sample Usage: python corner_detector.py checker.jpg 9 5 15000
    print >> sys.stderr, 'python '+__file__+' input_image denoising_sigma neighborhood_width threshold out_dir'
    sys.exit(1)

def main():
    if len(sys.argv) < 6:
        usage()
    print 
    
    # Get Params
    input_image_location = os.path.abspath(sys.argv[1])
    denoising_sigma = float(sys.argv[2]) # the sigma we use for our initial gaussian denoising of our image
    neighborhood_width = float(sys.argv[3]) # how big of a neighborhood we want to check for corners
    threshold = float(sys.argv[4]) # the threshold we want to use to make sure our our eigenalues are large enough
    out_dir = os.path.abspath(sys.argv[5]) # the output directory
    makedirs(out_dir)
    output_image_location = os.path.join(out_dir,'out.png') # the output image
    
    I = numpy.asarray(Image.open(input_image_location)).astype('float') # converting our input inmage values in the range [0,255] to float so that we don't lose precision
    I_h, I_w, I_channels = I.shape # get our input array dimensions
    
    total_run_time_start = time.time()
    
    # Filtered Gradient ##############################################################################    
    # Denoise with Gaussian Filter
    print "Denoising."
    denoise_start = time.time()
    denoising_kernel = get_gaussian_kernel(KERNEL_DIM, denoising_sigma) # get the gaussian kernel we're going to use for denoising
    I_denoised = convolve(I, denoising_kernel) # convolve with our kernel
    save_image(I,os.path.join(out_dir,'input.png'))
    save_image(I_denoised,os.path.join(out_dir,'denoised.png'))
    denoise_end = time.time()
    
    # Find x and y components of gradient
    find_gradient_start = time.time()
    I_denoised_grayscale = convert_to_grayscale(I_denoised) # convert our image to grayscale since we don't want to find different corners in all 3 color channels
    sobel_x = numpy.array([[-1, 0, 1], # our x sobel kernel
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype='float')
    sobel_y = numpy.array([[-1, -2, -1], # our y sobel kernel
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype='float')
    Fx = convolve(I_denoised_grayscale, sobel_x) # the gradient in the x direction
    Fy = convolve(I_denoised_grayscale, sobel_y) # the gradient in the y direction
    save_image(Fx,os.path.join(out_dir,'Fx.png'))
    save_image(Fy,os.path.join(out_dir,'Fy.png'))
    find_gradient_end = time.time()
    
    # Finding Corners ################################################################################
    #  Compute the covariance matrix C over a neighborhood around each point
    print "Detecting Corners."
    corner_detection_start = time.time()
    multiply_vectorized = numpy.vectorize(multiply) # I made a vectorized mult op
    Fx_squared = numpy.square(Fx) # made a squared matrix for each gradient direction
    Fy_squared = numpy.square(Fy)
    FxFy = multiply_vectorized(Fx,Fy)
    save_image(Fx_squared,os.path.join(out_dir,'Fx_squared.png'))
    save_image(Fy_squared,os.path.join(out_dir,'Fy_squared.png'))
    save_image(FxFy,os.path.join(out_dir,'FxFy.png'))
    
    ones_kernel = numpy.ones([neighborhood_width,neighborhood_width]) # a ones kernel that I'm using to take sums
    Sigma_Fx_squared = convolve(Fx_squared, ones_kernel, zero_borders=True) # in order to sum the square gradient values over a neighborhood, I just convolve the square matrix with a ones kernel that is the size of the desired neighborhood
    Sigma_Fy_squared = convolve(Fy_squared, ones_kernel, zero_borders=True) # I use zero values on the boundary to avoid having complications with edge cases (it's also more true to the sum if we just use zero valeus outside of the image
    Sigma_FxFy = convolve(FxFy, ones_kernel, zero_borders=True) # Similarly I'm taking the sum over Fx *Fy values
    
    save_image(Sigma_Fx_squared,os.path.join(out_dir,'Sigma_Fx_squared.png'))
    save_image(Sigma_Fy_squared,os.path.join(out_dir,'Sigma_Fy_squared.png'))
    save_image(Sigma_FxFy,os.path.join(out_dir,'Sigma_FxFy.png'))
    
    get_smaller_eigenvalue_vectorized = numpy.vectorize(get_smaller_eigenvalue) # see the top of this code for how this func works. It essentially just takes the three distinct sums we need for our 2x2 covariance matrix and returns the smaller eigen value. I vectorize this operation so that I can use it on the matrixes of sums I just calculated above (the Sigma_* values)
    
    smaller_eigenvalues = get_smaller_eigenvalue_vectorized(Sigma_Fx_squared,Sigma_Fy_squared,Sigma_FxFy) # get the smaller eigen values
    save_image(smaller_eigenvalues,os.path.join(out_dir,'smaller_eigenvalues.png'))
    
    thresholded_smaller_eigenvalues = numpy.array(smaller_eigenvalues) # threshold the eigen values to make sure they're all above the specified threshold.
    thresholded_smaller_eigenvalues[thresholded_smaller_eigenvalues<threshold] = 0 # zero out elements below the threshold
    save_image(thresholded_smaller_eigenvalues,os.path.join(out_dir,'thresholded_smaller_eigenvalues.png'))
    
    y_values, x_values = map(list, numpy.nonzero(thresholded_smaller_eigenvalues))
    corner_detection_end = time.time()
    
    # Nonmaximum Suppression #########################################################################
    print "Performing Nonmaximum Suppression."
    nonmaximal_suppression_start = time.time()
    L = sorted([(y,x,thresholded_smaller_eigenvalues[y,x]) for y,x in zip(y_values, x_values)], key=lambda e:e[2]) # list of potential corner positions
    
    def in_same_neighborhood(y_1,x_1,y_2,x_2): # func to tell if two potential corners (that specify a neighborhood) are overlapping
        return abs(y_1-y_2)<=neighborhood_width \
           and abs(x_1-x_2)<=neighborhood_width
    
    element_has_been_removed = True # this is inially set to true so that we can get the loop started
    while element_has_been_removed:
        element_has_been_removed = False # set to false because we haven't actually removed a point yet
        index_to_remove = None
        for i_1,(y_1,x_1,v_1) in enumerate(L): # here we iterate through each pair of points
            for i_2,(y_2,x_2,v_2) in enumerate(L):
                if in_same_neighborhood(y_1,x_1,y_2,x_2):
                    if v_1 > v_2: # if one point is bigger, we remove the small point
                        index_to_remove = i_2
                        element_has_been_removed = True
                        break
                    if v_1 < v_2:
                        index_to_remove = i_1
                        element_has_been_removed = True
                        break
            if element_has_been_removed: # we remove a point, we go back to the beginning of the while loop and start again because we don't want to iterate over a list that we're also manipulating, BAD THINGS COULD HAPPEN.
                break
        if element_has_been_removed:
            L.pop(index_to_remove)
    print str(len(L))+' corners found.' 
    nonmaximal_suppression_end = time.time()
    
    # Draw boxes
    print "Visualizing."
    visualizing_start = time.time()
    final_output = numpy.array(I)
    for (y,x,v) in L: 
        upper_y_outer = clamp(y-neighborhood_width,0,I_h)
        upper_y_inner = clamp(y-neighborhood_width+BOX_THICKNESS,0,I_h)
        left_x_outer  = clamp(x-neighborhood_width,0,I_w)
        left_x_inner  = clamp(x-neighborhood_width+BOX_THICKNESS,0,I_w)
        lower_y_outer = clamp(y+neighborhood_width,0,I_h)
        lower_y_inner = clamp(y+neighborhood_width-BOX_THICKNESS,0,I_h)
        right_x_outer = clamp(x+neighborhood_width,0,I_w)
        right_x_inner = clamp(x+neighborhood_width-BOX_THICKNESS,0,I_w)
        final_output[upper_y_outer:upper_y_inner,left_x_outer:right_x_outer,:] = BOX_COLOR # top line
        final_output[upper_y_outer:lower_y_outer,left_x_outer:left_x_inner,:] = BOX_COLOR # left line
        final_output[lower_y_inner:lower_y_outer,left_x_outer:right_x_outer,:] = BOX_COLOR # bottom line
        final_output[upper_y_outer:lower_y_outer,right_x_inner:right_x_outer,:] = BOX_COLOR # right line
    visualizing_end = time.time()
    
    total_run_time_end = time.time()
    # Save output image
    final_output = round_vectorized(final_output) # round our values so that we don't just truncate away information (we want int values since Python Image Library likes saving things in the range [0,255]
    final_output = clamp_array(final_output) # clamp our values to be in the range that Python Image Library likes, i.e. [0,255]
    final_output = final_output.astype('uint8') # convert it to the correct type so that Python Image Library won't complain
    Image.fromarray(final_output).save(output_image_location) # save our image
    
    # print performance information
    print 
    print "Canny Edge Detector Complete!"
    print "Denoising Time: "+str(denoise_end-denoise_start)+" seconds"
    print "Gradient Calculation Time: "+str(find_gradient_end-find_gradient_start)+" seconds"
    print "Corner Detection Time: "+str(corner_detection_end-corner_detection_start)+" seconds"
    print "Nonmaximal Suppression Time: "+str(nonmaximal_suppression_end-nonmaximal_suppression_start)+" seconds"
    print "Visualization Time: "+str(visualizing_end-visualizing_start)+" seconds"
    print "Total Run Time: "+str(total_run_time_end-total_run_time_start)+" seconds"
    print 

if __name__ == '__main__':
    main()


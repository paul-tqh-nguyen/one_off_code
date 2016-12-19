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

KERNEL_DIM = 13 # we're using a pretty big finite kernel size just because our sigmas can get pretty large

def usage():
    # Sample Usage: python sift.py building.jpg 4 1.6 3 0.003
    print >> sys.stderr, 'python '+__file__+' input_image num_octaves min_sigma num_intervals_per_octave low_contrast_threshold out_dir'
    sys.exit(1)

def main():
    if len(sys.argv) < 7:
        usage()
    print 
    
    # Get Params
    input_image_location = os.path.abspath(sys.argv[1]) # get the name of our input image
    num_octaves = int(sys.argv[2]) # get the number of octaves we want
    min_sigma = float(sys.argv[3]) # get our base starting sigma value for our gaussian pyramid
    num_intervals_per_octave = int(sys.argv[4]) # referred to as s in the paper
    low_contrast_threshold = float(sys.argv[5]) # this is the threshold we use to remove points with low contrast during the localization stage
    out_dir = os.path.abspath(sys.argv[6]) # the output directory
    makedirs(out_dir)
    output_image_location = os.path.join(out_dir,'out.png') # the output image
    
    I = numpy.asarray(Image.open(input_image_location)).astype('float') # open our input image
    save_image(I,os.path.join(out_dir,'input.png'))
    I /= 255.0 # the paper discussed images in the range [0.0,1.0], so I also choose to work in this range
    I_h, I_w, I_channels = I.shape # get the dimensions of our input image
    
    total_run_time_start = time.time()
    
    # Scale-space extrema detection ##################################################################    
    # Calculate DoG images
    print "Calculating DoG images."
    DoG_start = time.time()
    k = 2**(1.0/num_intervals_per_octave) # calculate the base factor k which separates our gaussian pyramids, this value can be derived by hand (which is what I did as a sanity check), the formula is also shown in the paper
    I_grayscale = convert_to_grayscale(I) # we're converting our image to grayscale since we check for edges in gray scale
    DoG_images_by_octave = []
    for octave_index in range(num_octaves): # calculate all the guassian images for one octave
        blurred_images = []
        for scale_index in range(num_intervals_per_octave+3): # go through all of the sigmas we need to get s valid DoG images
            sigma = min_sigma*(2.0**octave_index)*(k**scale_index) # calculate the sigma from the index
            kernel = get_gaussian_kernel(KERNEL_DIM, sigma) # get the kernel from the sigma
            I_blurred = convolve(I_grayscale, kernel) # convolve with the gaussian kernel
            blurred_images.append( I_blurred ) # we stick this blurred image in our list
        DoG_images_for_this_octave = []
        for scale_index in range(num_intervals_per_octave+2): # here we calculate all of our DoG images
            DoG_images_for_this_octave.append(blurred_images[scale_index]-blurred_images[scale_index+1]) # just sbtract our blurred images 
        DoG_images_by_octave.append(DoG_images_for_this_octave) # we stick the DoG images for this octave in our list
        I_grayscale = downsample_2d(I_grayscale) # downsample our image for the next octave
    DoG_end = time.time() 
    
    # Find extrema 
    print "Finding Extrema."
    extrema_start = time.time()
    candidate_points_0 = set()
    for octave_index, DoG_image_list in enumerate(DoG_images_by_octave): # for each octave
        for DoG_image_index in xrange(1,len(DoG_image_list)-1): # for each DoG image / scale (except the top and bottome ones since they are border cases)
            DoG_image = DoG_image_list[DoG_image_index] # we nab our DoG image
            height, width = DoG_image.shape[0:2] # we get it's heigh and width
            for y in xrange(1,height-1): # we iterate through all potential extrema points
                for x in xrange(1,width-1): # (except the edge points)
                    low_y  = y-1 # we're getting the surrounding point indices here
                    low_x  = x-1
                    high_y = y+1
                    high_x = x+1
                    low_DoG_image_index  = DoG_image_index-1
                    high_DoG_image_index = DoG_image_index+1
                    current_val = DoG_image_list[DoG_image_index][y,x] # get the value of the potential extrema point (the one in the middle)
                    is_max = True # we assume it's the biggest / smalled of the surrounding points we 
                    is_min = True # have checked so far until proven otherwise
                    for DoG_image_index_prime in xrange(low_DoG_image_index,high_DoG_image_index+1):
                        for y_prime in xrange(low_y,high_y+1):
                            for x_prime in xrange(low_x,high_x+1):
                                if current_val < DoG_image_list[DoG_image_index_prime][y_prime,x_prime]:
                                    is_max = False
                                if current_val > DoG_image_list[DoG_image_index_prime][y_prime,x_prime]:
                                    is_min = False
                                if (not is_max) and (not is_min): # if we know it's not the min or max, we quit ASAP and go to the next potential extrema point
                                    break
                            if (not is_max) and (not is_min):
                                break
                        if (not is_max) and (not is_min):
                            break
                    if is_max or is_min: # if it's an extrema, we add it to our list of potential candidates 
                        candidate_points_0.add( (y,x,octave_index,DoG_image_index) )
    extrema_end = time.time()
    print str(len(candidate_points_0))+' candidate points found.'
    
    # Localization
    localization_start = time.time()
    candidate_points_1 = []
    for (y,x,octave_index,DoG_image_index) in candidate_points_0: # iterate through all of our extrema points
        DoG_image_list = DoG_images_by_octave[octave_index]
        DoG_image = DoG_image_list[DoG_image_index]
        
        # calculate the matrix content values
        df_dx = (DoG_image_list[DoG_image_index][y,x+1] - DoG_image_list[DoG_image_index][y,x-1])/2.0
        df_dxx = DoG_image_list[DoG_image_index][y,x+1] + DoG_image_list[DoG_image_index][y,x-1] - 2.0 * DoG_image_list[DoG_image_index][y,x]
        df_dy = (DoG_image_list[DoG_image_index][y+1,x] - DoG_image_list[DoG_image_index][y-1,x])/2.0
        df_dyy = DoG_image_list[DoG_image_index][y+1,x] + DoG_image_list[DoG_image_index][y-1,x] - 2.0 * DoG_image_list[DoG_image_index][y,x]
        df_dxy = (DoG_image_list[DoG_image_index][y+1,x+1]+DoG_image_list[DoG_image_index][y-1,x-1]-DoG_image_list[DoG_image_index][y-1,x+1]-DoG_image_list[DoG_image_index][y+1,x-1])/4.0
        df_ds = (DoG_image_list[DoG_image_index+1][y,x] - DoG_image_list[DoG_image_index-1][y,x])/2.0
        df_dss = DoG_image_list[DoG_image_index+1][y,x] + DoG_image_list[DoG_image_index-1][y,x] - 2.0 * DoG_image_list[DoG_image_index][y,x]
        df_dsx = (DoG_image_list[DoG_image_index+1][y,x+1]+DoG_image_list[DoG_image_index-1][y,x-1]-DoG_image_list[DoG_image_index-1][y,x+1]-DoG_image_list[DoG_image_index+1][y,x-1])/4.0
        df_dsy = (DoG_image_list[DoG_image_index+1][y+1,x]+DoG_image_list[DoG_image_index-1][y-1,x]-DoG_image_list[DoG_image_index-1][y+1,x]-DoG_image_list[DoG_image_index+1][y-1,x])/4.0
        
        b = numpy.array( [ df_dx,
                           df_dy,
                           df_ds ] , dtype='float')        
        A = numpy.array( [ # the actual matrix
                           [df_dxx,df_dxy,df_dsx],
                           [df_dxy,df_dyy,df_dsy],
                           [df_dsx,df_dsy,df_dss] 
                         ] , dtype='float')
        
        X_hat = numpy.dot(numpy.linalg.pinv(A),b) # solve the matrix equation
        
        if X_hat[0] > 0.5: # the paper said that values > 0.5 in any direction would mean that the offset was significant, so that's what I do here.
            x_offset = 1
        elif X_hat[0] < -0.5:
            x_offset = -1
        else:
            x_offset = 0
        
        if X_hat[1] > 0.5:
            y_offset = 1
        elif X_hat[1] < -0.5:
            y_offset = -1
        else:
            y_offset = 0
        
        if X_hat[2] > 0.5:
            s_offset = 1
        elif X_hat[2] < -0.5:
            s_offset = -1
        else:
            s_offset = 0
        
        if DoG_image_index+s_offset in [0,len(DoG_image_list)-1] or y+y_offset in [0,DoG_image.shape[0]] or x+x_offset in [0,DoG_image.shape[1]]: # if the offset pushes the point to an invalid spot, i.e. a boundary position, we throw it away since boundary features are hard to match (and also because the paper said to)
            continue
        if abs(DoG_image_list[DoG_image_index+s_offset][y+y_offset,x+x_offset]) > low_contrast_threshold: # Low contrast rejection
            r = 10
            R = (df_dxx+df_dyy)**2 / (df_dxx*df_dyy-df_dxy**2) # jsut followed the formula from the paper
            if R < (r+1)**2/r: # Eliminating edge responses
                candidate_points_1.append( (y+y_offset,x+x_offset,octave_index,DoG_image_index+s_offset) )
    print str(len(candidate_points_1))+' final keypoints.'
    localization_end = time.time()
    
    # Visualize output
    visualize_start = time.time()
    line_color = [255,0,0] # we visualize our key points in red
    line_thickness = 1
    final_output = numpy.array(I*255.0)
    for (y,x,octave_index,DoG_image_index) in candidate_points_1:
        scale = min_sigma*(2**octave_index)*(k**DoG_image_index) # we use the smaller sigma of the gaussians used to calculate the DoG as the radius for the red circle we draw around our sift  points
        box_width = scale**2.5
        for angle_degrees in xrange(0,360,5): # we draw circles around the sift points where the radius is proportional to the 
            angle = angle_degrees*math.pi/180.0
            if x+box_width*math.cos(angle) < I_w and x+box_width*math.cos(angle) >= 0 and y+box_width*math.sin(angle) < I_h and y+box_width*math.sin(angle) >= 0:
                final_output[y+box_width*math.sin(angle),x+box_width*math.cos(angle),:] = line_color 
    final_output = clamp_array(final_output)
    final_output = final_output.astype('uint8')
    Image.fromarray(final_output).save(output_image_location)
    visualize_end = time.time()
    
    total_run_time_end = time.time()
    
    # print performance data
    print 
    print "SIFT Complete!"
    print "DoG Time: "+str(DoG_end-DoG_start)+" seconds"
    print "Extrema Time: "+str(extrema_end-extrema_start)+" seconds"
    print "Localization Time: "+str(localization_end-localization_start)+" seconds"
    print "Visualization Time: "+str(visualize_end-visualize_start)+" seconds"
    print "Total Run Time: "+str(total_run_time_end-total_run_time_start)+" seconds"
    print 

if __name__ == '__main__':
    main()


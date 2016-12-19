#!/usr/bin/python

'''
'''

import sys
import os
import pdb

def main():
    html_code = '''

<head><title>CS 4501/6501: Programming Assignment 2 by Paul Nguyen (pn2yr@virginia.edu)</title></head>

<h2> 
Paul Nguyen (pn2yr@virginia.edu)
<BR>
<BR>
Assignment 1: Edge and feature detector
<BR>

<BR> Code: 
<BR> <a href="canny_edge_detector.py"> Canny Edge Detector (edge_detector.py) </a>
<BR> <a href="corner_detector.py"> Corner Detector (corner_detector.py) </a>
<BR> <a href="sift.py"> SIFT (sift.py)</a>
<BR> <a href="util.py"> Helper Functions (util.py, necessary to get any of the above code running) </a>
<BR>
</h2>

<hr noshade color="#000000">

<h3>0. Intro </h3>
I'm using one of my late days to submit this assignment to submit this late on Feb 6.

<BR>
<BR>
'''
    def write():
        with open('WRITEUP.html','wt') as fff:
            fff.write(html_code)
    
    # Generate Edge Detector Results
    # Canny Edge Detector Gen Start ################################################################################
    canny_edge_detector_supplemental_html_file_name = 'canny_edge_detector_supplemental.html'
    html_code += '''

<hr noshade color="#000000">

<h3>1. Canny edge detector </h3>

<h4> Usage </h4>

<strong>Usage</strong>: python canny_edge_detector.py input_image denoising_sigma low_threshold high_threshold out_dir<BR><BR>

<strong>Sample Usage</strong>: python canny_edge_detector.py building.jpg 9 35 75 ~/Desktop/results/<BR><BR>

<strong>input_image</strong>: The image we want to detect edges for. <BR>
<strong>denoising_sigma</strong>: The sigma for determining the coefficients of the Gaussian kernel used for denoising the image. <BR>
<strong>low_threshold</strong>: The lower threshold to be used for hysteresis thresholding. <BR>
<strong>high_threshold</strong>: The higher threshold to be used for hysteresis thresholding.<BR>
<strong>out_dir</strong>: The directory where results are placed. The edges are shown in out.png. The denoised image is shown in denoised.png. The horizontal gradient is shown in Fx.png. The vertical gradient is shown in Fy.png. The magnitudes of the gradients are shown in magnitude.png. The nonmaximum suppressed gradient magnitudes are shown in suppressed_magnitude.png. See our results for examples of these images. <BR>


<h4> Implementation Details </h4>
My edge detector uses integer values in the range [0,255] to represent intensity values, and the values I use to get certain results reflects that. The parameters would be different if my code represented intensities as floats in [0.0,1.0]. <BR><BR>
The parameters of the edge detector are the sigma to be used to determine the Gaussian kernel for denoising, the low threshold for the hysteresis thresholding, and the high threshold for the hysteresis thresholding. <BR><BR>
I first determine the Gaussian kernel using the formulae from the slides. Since Gaussians have a non-zero value for all inputs, I approximate the Gaussian kernel using a finite 5x5 kernel. I just normalize the finite kernel after I calculate the actual values in those 5x5 positions. See util.py for further details on how I implemented the blurring. I used a built in Python function to actually convolve. At the edges, I just used the values nearest to the point being inquired about since using zero values would in most cases lead to visible artifacts that we don't want to introduce into an image (especially for an edge detector). <BR><BR>
I used the Sobel Filter to calculate the gradients in the horizontal and vertical directions. <BR><BR>
I computed the magnitudes and orientations of the gradients using methods described in class and using NumPy's vectorized operations. 
I quantized the orientations to multiples of 45 degrees and performed my nonmaximum suppression using these quantized values.
Hysteresis thresholding was the final step in edge detection. I created a final answer buffer of zeros. I copied into this buffer the edge strengths at points where the edge strength was higher than the high threshold. I then added the 8 surrounding neighbors of these points to a "to-do list". I marked these points as visited. I then went through this to-do list checking if any of them were above the low threshold. I mark all of these points as visited along the way. If any were above the low threshold, I added them to the final solution buffer and added all of its neighbors to the to-do list and would remove the point from the to-do list. The program terminates when the to-do list is empty. Then we have our final answer. 
<BR> 

<h4> Discussion </h4>
The Gaussian kernel used for denoising plays a huge factor in the results we get. The blurring obviously removes noise, but it also helps determine which edges are most significant. If an edge is still present and detected after an initial denoising Gaussian blur with a huge sigma (which means a lot of blurring), then that edge is clearly very strong. The nonmaximum suppession tells us where that edge is located, but the Gaussian blur is what impacts which edges are shown in our final result. The low and high thresholds are also used to control which edges are shown. The high threshold determines which edges are strong enough to be put in our final answer. The low threshold helps us limit the edge to just being the very strong part of the edge (it's possible that an edge might very long, but only part of the edge has a high gradient magnitude). 

<BR>

<h4> Results </h4>
Particularly interesting results are shown <a href="canny_edge_detector_results/corner_detector[image:building.jpg][sigma:9][low:30][high:70]/sub_link.html">here</a>, <a href="canny_edge_detector_results/corner_detector[image:building.jpg][sigma:30][low:40][high:70]/sub_link.html">here</a>, and <a href="canny_edge_detector_results/corner_detector[image:mandrill.jpg][sigma:30][low:20][high:60]/sub_link.html">here</a>. <BR><BR>

We present supplemental results for various denoising Gaussian kernels, low thresholds, and high thresholds <a href="'''+canny_edge_detector_supplemental_html_file_name+'''">here</a>. 

'''
    canny_edge_detector_supplemental_html_code = '''

<head><title>CS 4501/6501: Programming Assignment 1 by Paul Nguyen (pn2yr@virginia.edu)</title></head>

<h2> 
Paul Nguyen (pn2yr@virginia.edu)
<BR>
<BR>
Canny Edge Detector
<BR>
</h2>

<BR> Code: 
<BR> <a href="canny_edge_detector.py"> Canny Edge Detector</a>
<BR> <a href="corner_detector.py"> Corner Detector</a>
<BR> <a href="sift.py"> SIFT </a>
<BR>

<hr noshade color="#000000">

<BR>
We present supplemental results for various denoising Gaussian kernels, low thresholds, and high thresholds below.

<BR>
<BR>
'''
    edge_dectector_output_location = os.path.relpath('./canny_edge_detector_results')
    makedirs(edge_dectector_output_location)
    sigmas = [9,15,30]
    low_thresholds = [10, 20, 30, 40, 50, 60, 70, 80] 
    high_thresholds = [20, 30, 40, 50, 60, 70, 80, 90] 
    images = ['building.jpg','mandrill.jpg']
    
#    sigmas = sigmas[:1]
#    low_thresholds = low_thresholds[:1]
#    high_thresholds = high_thresholds[:1]
#    images = images[:1]
    
    job_count = 0
    commands = ''
    for image in images:
        for sigma in sigmas:
            canny_edge_detector_supplemental_html_code += '''<BR>
Image: '''+image+''' [&sigma;:'''+str(sigma)+'''] <BR>
'''
            canny_edge_detector_supplemental_html_code += '''
<table border="3" style='width:1500px' class="fixed">
'''
            canny_edge_detector_supplemental_html_code += '''<tr> 
  <td align="center" valign="center">
  </td>'''
            for high_threshold in high_thresholds:
                canny_edge_detector_supplemental_html_code += '''
  <td align="center" valign="center">
    <font size="3">
      '''+' [high:'+str(high_threshold)+']'+'''
    </font>
  </td>
'''
            canny_edge_detector_supplemental_html_code += '''</tr>'''
            for low_threshold in low_thresholds:
                canny_edge_detector_supplemental_html_code += '''<tr> '''
                canny_edge_detector_supplemental_html_code += '''
  <td align="left" valign="center">
    <font size="3">
      '''+'[low:'+str(low_threshold)+']'+'''
    </font>
  </td>
'''
                for high_threshold in high_thresholds:
                    if low_threshold < high_threshold:
                        out_dir = os.path.join(edge_dectector_output_location,'corner_detector[image:'+str(image)+'][sigma:'+str(sigma)+'][low:'+str(low_threshold)+'][high:'+str(high_threshold)+']')
                        makedirs(out_dir)
                        sub_html_file_name = os.path.join(out_dir,'sub_link.html')
                        with open(sub_html_file_name,'wt') as f:
                            f.write('''
<h3> Canny edge detector </h3> <BR>
<hr noshade color="#000000">
<h4> Denoising Sigma: '''+str(sigma)+''' </h4>
<h4> Low Threshold: '''+str(low_threshold)+''' </h4>
<h4> High Threshold: '''+str(high_threshold)+''' </h4>

<BR><BR><BR><BR>
<h4> Input Image: </h4> <img src="input.png"> <BR> <BR> <BR>
<h4> Denoised Image: </h4> <img src="denoised.png"> <BR> <BR> <BR>
<h4> Horizontal Gradient: </h4> <img src="Fx.png"> <BR> <BR> <BR>
<h4> Vertical Gradient: </h4> <img src="Fy.png"> <BR> <BR> <BR>
<h4> Gradient Magnitude: </h4> <img src="magnitude.png"> <BR> <BR> <BR>
<h4> Suppressed Gradient Magnitude: </h4> <img src="suppressed_magnitude.png"> <BR> <BR> <BR>
<h4> Final Edges: </h4> <img src="out.png"> <BR> <BR> <BR>
''')
                        image_link_name = os.path.join(out_dir,'out.png')
                        if not os.path.isfile(image_link_name):
                            commands += ('python canny_edge_detector.py '+str(image)+' '+str(sigma)+' '+str(low_threshold)+' '+str(high_threshold)+' '+str(out_dir)+' & ')
                            job_count+= 1
                        if job_count == 4:
                            job_count = 0
                            system('('+commands+'wait'+') > /dev/null 2>&1')
                            print 
                            commands = ''
                        canny_edge_detector_supplemental_html_code += '''
  <td align="left" valign="center">
    <font size="3">
      '''+' <a href="'+sub_html_file_name+'">[low:'+str(low_threshold)+'][high:'+str(high_threshold)+']</a>'+'''
    </font>
  </td>
'''
                    else:
                        canny_edge_detector_supplemental_html_code += '''
  <td align="left" valign="center">
    <font size="3">  </font>
  </td>
'''
                canny_edge_detector_supplemental_html_code += '''</tr>'''
            canny_edge_detector_supplemental_html_code += '''
</table>
'''
    system('('+commands+'wait'+') > /dev/null 2>&1')
    with open(canny_edge_detector_supplemental_html_file_name,'wt') as f:
        f.write(canny_edge_detector_supplemental_html_code)
    # Canny Edge Detector Gen End ##################################################################################
    
    # Harris Corner Detector Gen Start #############################################################################
    corner_detector_supplemental_html_file_name =  'corner_detector_supplemental.html'
    html_code += '''

<hr noshade color="#000000">

<h3>2. Harris Corner detector </h3>

<h4> Usage </h4>

<strong>Usage</strong>: python corner_detector.py input_image denoising_sigma neighborhood_width threshold out_dir <BR><BR>

<strong>Sample Usage</strong>: python corner_detector.py checker.jpg 9 5 15000 ~/Desktop/results/<BR><BR>

<strong>input_image</strong>: The image we want to detect corners for. <BR>
<strong>denoising_sigma</strong>: The sigma for determining the coefficients of the Gaussian kernel used for denoising the image. <BR>
<strong>neighborhood_width</strong>: The lower threshold to be used for hysteresis thresholding. <BR>
<strong>threshold</strong>: The higher threshold to be used for hysteresis thresholding.<BR>
<strong>out_dir</strong>: The directory where results are placed. The corners are shown in out.png (corners are shown in blue boxes). The box sizes correspond to the the neighborhood widths. The denoised image is shown in denoised.png. The horizontal gradient is shown in Fx.png. The vertical gradient is shown in Fy.png. The values of the squared horizontal gradients used for our covariance matrix are shown in Fx_squared.png. The values of the squared vertical gradients used for our covariance matrix are shown in Fy_squared.png. The values of the product of the vertical gradients and the horizontal gradients used for our covariance matrix are shown in FxFy.png. The sums over neighborhoods around each point of our squared horizontal gradients, squared vertical gradients, and product of vertical gradients and the horizontal gradients used in our covariance matrix are shown in Sigma_Fx_squared.png, Sigma_Fy_squared.png, and Sigma_FxFy.png, respectively. The smaller eigenvalues of our covariance matrices are shown in smaller_eigenvalues.png. The smaller eigenvalues that survive thresholding are shown in thresholded_smaller_eigenvalues.png. See our results for examples of these images. <BR>


<h4> Implementation Details </h4>
I denoise and find gradients in the same manner as I have in the Canny edge detector. <BR><BR>
The covariance matrix we use for determining if a point is a corner contains values of the squared sum over the horizontal gradients surrounding the point, the squared sum over the vertical gradients surrounding the point, and the product of sum over the vertical gradients with the sum over the horizontal gradients surrounding the point. I first calculated the horizontal and vertical gradients. I then calculated the squares of these gradients and the product of them (the array multiplication, not matrix multiplication). In order to get the sum of the squared horizontal or squared vertical gradients over the neighborhood of a point, I just convolved the squared horizontal and squared vertical gradients with a matrix of ones that has the desired neighborhood width. I then wrote a function that if given the squared sum over the horizontal gradients surrounding a point, the squared sum over the vertical gradients surrounding a point, and the product of sum over the vertical gradients with the sum over the horizontal gradients surrounding a point, it would return the smaller eigen value of the covariance matrix formed by these three values. I vectorized this function to work with matrices of values and applied it to the matrices of squared gradients I determined earlier. I now have a matrix of smaller eigen values. <BR><BR>
I put all the smaller eigen values that survived thresholding along with their coordinates into a list. <BR><BR>
I then performed nonmaximum suppession. I naively in quadratic time checked each pair of points to see if they were in the same neighborhood and removed the smaller of the two. I kept doing this until no more points were within the same neighborhood. <BR><BR>
<BR> 

<h4> Discussion </h4>
The neighborhood width, which determined the window size of a potential corner, plays a big role in which points would be considered corners. A small neighborhood size would be accurately find corners, but not necessarily ones we care about, e.g. if our window size is really small, we may detect a bunch of corners along blades of grass in an image, but ignore larger corners like those found on a door frame in the same image. This is where the denoising sigma can also play a big role in our outcome. A sufficiently large denoising sigma can remove "less important" corners. The eigenvalue threshold is something that we use to remove corners that may not be sharp enough. We usually determine this after we determine the window size and denoising sigma for a particular image so that we can determine how strong of corners we want. We use this parameter to help weed out the "less important" potential corners.

<BR>

<h4> Results </h4>
Particularly interesting results are shown <a href="corner_detector_results/corner_detector[image:checker.jpg][sigma:15][threshold:15000][width:5]/sub_link.html">here</a>. <BR><BR>

We present supplemental results for various denoising Gaussian kernels, neighborhood sizes, and thresholds <a href="'''+corner_detector_supplemental_html_file_name+'''">here</a>. <BR><BR>

<BR>

'''
    corner_detector_supplemental_html_code = '''

<head><title>CS 4501/6501: Programming Assignment 1 by Paul Nguyen (pn2yr@virginia.edu)</title></head>

<h2> 
Paul Nguyen (pn2yr@virginia.edu)
<BR>
<BR>
Harris Corner Detector
<BR>
</h2>

<BR> Code: 
<BR> <a href="canny_edge_detector.py"> Canny Edge Detector</a>
<BR> <a href="corner_detector.py"> Corner Detector</a>
<BR> <a href="sift.py"> SIFT </a>
<BR>

<hr noshade color="#000000">

<BR>
We present supplemental results for various denoising Gaussian kernels, thresholds, and neighborhood sizes below.

<BR>
<BR>
'''
    corner_detector_output_location = os.path.relpath('./corner_detector_results')
    makedirs(corner_detector_output_location)
    sigmas = [9,15,30]
    neighborhood_widths = [5]
    thresholds = [10000, 15000, 20000, 25000, 30000] 
    images = ['building.jpg','mandrill.jpg', 'checker.jpg']
    
#    sigmas = sigmas[:1]
#    neighborhood_widths = neighborhood_widths[:1]
#    thresholds = thresholds[:1]
#    images = images[:1]
    
    job_count = 0
    commands = ''
    for image in images:
        for sigma in sigmas:
            corner_detector_supplemental_html_code += '''<BR>
Image: '''+image+''' [&sigma;:'''+str(sigma)+'''] <BR>
'''
            corner_detector_supplemental_html_code += '''
<table border="3" style='width:1500px' class="fixed">
'''
            corner_detector_supplemental_html_code += '''<tr> 
  <td align="center" valign="center">
  </td>'''
            for neighborhood_width in neighborhood_widths:
                corner_detector_supplemental_html_code += '''
  <td align="center" valign="center">
    <font size="3">
      '''+' [width:'+str(neighborhood_width)+']'+'''
    </font>
  </td>
'''
            corner_detector_supplemental_html_code += '''</tr>'''
            for threshold in thresholds:
                corner_detector_supplemental_html_code += '''<tr> '''
                corner_detector_supplemental_html_code += '''
  <td align="left" valign="center">
    <font size="3">
      '''+'[threshold:'+str(threshold)+']'+'''
    </font>
  </td>
'''
                for neighborhood_width in neighborhood_widths:
                    out_dir = os.path.join(corner_detector_output_location,'corner_detector[image:'+str(image)+'][sigma:'+str(sigma)+'][threshold:'+str(threshold)+'][width:'+str(neighborhood_width)+']')
                    makedirs(out_dir)
                    sub_html_file_name = os.path.join(out_dir,'sub_link.html')
                    with open(sub_html_file_name,'wt') as f:
                        f.write('''
<h3> Harris Corner detector </h3> <BR>
<hr noshade color="#000000">
<h4> Denoising Sigma: '''+str(sigma)+''' </h4>
<h4> Threshold: '''+str(threshold)+''' </h4>
<h4> Neighborhood Width: '''+str(neighborhood_width)+''' </h4>

<BR><BR><BR><BR>
<h4> Input Image: </h4> <img src="input.png"> <BR> <BR> <BR>
<h4> Denoised Image: </h4> <img src="denoised.png"> <BR> <BR> <BR>
<h4> Horizontal Gradient: </h4> <img src="Fx.png"> <BR> <BR> <BR>
<h4> Vertical Gradient: </h4> <img src="Fy.png"> <BR> <BR> <BR>
<h4> Squared Horizontal Gradient: </h4> <img src="Fx_squared.png"> <BR> <BR> <BR>
<h4> Squared Vertical Gradient: </h4> <img src="Fy_squared.png"> <BR> <BR> <BR>
<h4> Horizontal Gradient x Vertical Gradient: </h4> <img src="FxFy.png"> <BR> <BR> <BR>
<h4> Sum of Squared Horizontal Gradient over Neighborhood: </h4> <img src="Sigma_Fx_squared.png"> <BR> <BR> <BR>
<h4> Sum of Squared Vertical Gradient over Neighborhood: </h4> <img src="Sigma_Fy_squared.png"> <BR> <BR> <BR>
<h4> Sum of Horizontal Gradient x Vertical Gradient over Neighborhood: </h4> <img src="Sigma_FxFy.png"> <BR> <BR> <BR>
<h4> Smaller Eigevalues: </h4> <img src="smaller_eigenvalues.png"> <BR> <BR> <BR>
<h4> Thresholded Smaller Eigevalues: </h4> <img src="thresholded_smaller_eigenvalues.png"> <BR> <BR> <BR>
<h4> Final Corners: </h4> <img src="out.png"> <BR> <BR> <BR>
''')
                    image_link_name = os.path.join(out_dir,'out.png')
                    if not os.path.isfile(image_link_name):
                        commands += ('python corner_detector.py '+str(image)+' '+str(sigma)+' '+str(neighborhood_width)+' '+str(threshold)+' '+str(out_dir)+' & ')
                        job_count+= 1
                    if job_count == 4:
                        job_count = 0
                        system('('+commands+'wait'+') > /dev/null 2>&1')
                        print 
                        commands = ''
                    corner_detector_supplemental_html_code += '''
  <td align="left" valign="center">
    <font size="3">
      '''+' <a href="'+sub_html_file_name+'">[threshold:'+str(threshold)+'][width:'+str(neighborhood_width)+']</a>'+'''
    </font>
  </td>
'''
                corner_detector_supplemental_html_code += '''</tr>'''
            corner_detector_supplemental_html_code += '''
</table>
'''
    system('('+commands+'wait'+') > /dev/null 2>&1')
    with open(corner_detector_supplemental_html_file_name,'wt') as f:
        f.write(corner_detector_supplemental_html_code)
    # Harris Corner Detector Gen End ###############################################################################
    
    # SIFT Gen Start ###############################################################################################
    sift_supplemental_html_file_name =  'sift_supplemental.html'
    html_code += '''

<hr noshade color="#000000">

<h3>3. SIFT </h3>


<h4> Usage </h4>

<strong>Usage</strong>: python sift.py input_image num_octaves min_sigma num_intervals_per_octave low_contrast_threshold out_dir <BR><BR>

<strong>Sample Usage</strong>: python sift.py building.jpg 4 1.6 3 0.003 ~/Desktop/results/ <BR><BR>

<strong>input_image</strong>: The image we want to detect edges for. <BR>
<strong>num_octaves</strong>: The number of octaves to have in our stack of DoG images. <BR>
<strong>min_sigma</strong>: The base sigma to be used to determine our stack of DoG images. <BR>
<strong>num_intervals_per_octave</strong>: The number of intervals, which determines the number of DoG images, in each octave. <BR>
<strong>low_contrast_threshold</strong>: The threshold to weed out unstable extrema points. <BR>
<strong>out_dir</strong>: Output directory. <BR>

<h4> Implementation Details </h4>
I first determine the value for k, the factor which separates the sigmas for each level of our Gaussian stack, using the formula described in the paper. I make a list where the indices correspond to an octave. Each element of this list is a list of the DoG images we calculated for the octave using k. <BR>
To find potential extrema, I iterate through the octaves and through the list of DoG images that corresponds to each octave and check whether or not any pixel in the current DoG image is the min or max of its 3x3x3 neighborhood in our stack of DoG images. If so, we consider it a candidate point. <BR>
We then localize the candidate points. We do this by following equation (3) of the paper. I followed the description described <a href="https://d1b10bmlvqabco.cloudfront.net/attach/i4q0xk6r2oh24d/hzj7zadl2dj5um/i5rsh8q18ril/siftppt1.ppt">here</a> to help find the derivatives needed to calculate our offset. If the DoG value at our point+offset is less than our low_contrast_threshold, we assert that the point is unstable and don't consider it. <BR>
The corner-detector-like testing is then performed like it is according to the paper, but since we're only comparing ratios of values, this runs much more quickly than the Harris corner detector that has to find eigen values. <BR>
Thus, we have our key points. I drew red circles around our keypoints. The radius of the circle is proportional to the scale in our stack of DoG images the keypoint was found in. <BR>

<h4> Discussion </h4>
The number of octaves we select gives us more candidate points, but after a certain point in our stack of DoG images, the Gaussian kernel becomes very large and most of the blurred images will differ very slightly. The number of intervals we have in each octave plays a significantly role in our final result since more intervals would lead us to have more DoG images, which would lead to more candidate points. However, we do not always get more candidate points since these points are not guaranteed to be extrema. Increasing this number also significantly increases computation time, so that's something very important to take into account. The low contrast threshold plays a significant role in helping us select how stable of key points we want. 
<BR>

<h4> Results </h4>

Particularly interesting results are shown <a href="sift_results/sift[image:building.jpg][num_octaves:4][num_octaves:4][base_sigma:1.6][num_intervals_per_octave:3][low_contrast_threshold:0.003]/sub_link.html">here</a> and <a href="sift_results/sift[image:checker.jpg][num_octaves:4][num_octaves:4][base_sigma:1.6][num_intervals_per_octave:5][low_contrast_threshold:0.006]/sub_link.html">here</a>. <BR><BR>

We present supplemental results for various numbers of octaves, base sigma values used for Gaussian blurring, numbers of intervals per octave, and low contrast thresholds <a href="'''+sift_supplemental_html_file_name+'''">here</a>. <BR><BR>

<h4> Results </h4>

<BR>

'''
    sift_supplemental_html_code = '''

<head><title>CS 4501/6501: Programming Assignment 1 by Paul Nguyen (pn2yr@virginia.edu)</title></head>

<h2> 
Paul Nguyen (pn2yr@virginia.edu)
<BR>
<BR>
SIFT
<BR>
</h2>

<BR> Code: 
<BR> <a href="canny_edge_detector.py"> Canny Edge Detector</a>
<BR> <a href="corner_detector.py"> Corner Detector</a>
<BR> <a href="sift.py"> SIFT </a>
<BR>

<hr noshade color="#000000">

<BR>
We present supplemental results for various numbers of octaves, base sigma values used for Gaussian blurring, numbers of intervals per octave, and low contrast thresholds  below.

There were several parameters, so it was difficult to organize the links in a meaningful way like I did for the Canny edge detector and Harris corner detector, so I instead just listed the outputs of some sets of parameters.

<BR>
<BR>
'''
    sift_output_location = os.path.relpath('./sift_results')
    makedirs(sift_output_location)
    num_octaves_list = range(4,10)
    base_sigmas = [1.1,1.6, 2.0,2.5,3]
    num_intervals_per_octave_list = range(3,6)
    low_contrast_thresholds = [0.003, 0.006, 0.009, 0.012, 0.015]
    images = ['building.jpg','mandrill.jpg', 'checker.jpg']
    
#    num_octaves_list = num_octaves_list[:1]
#    base_sigmas = base_sigmas[:1]
#    num_intervals_per_octave_list = num_intervals_per_octave_list[:1]
#    low_contrast_thresholds = low_contrast_thresholds[:1]
#    images = images[:1]
    
    job_count = 0
    commands = ''
    
    for num_octaves in num_octaves_list:
        for base_sigma in base_sigmas:
            for num_intervals_per_octave in num_intervals_per_octave_list:
                for low_contrast_threshold in low_contrast_thresholds:
                    for image in images:
                        out_dir = os.path.join(sift_output_location,'sift[image:'+str(image)+'][num_octaves:'+str(num_octaves)+'][num_octaves:'+str(num_octaves)+'][base_sigma:'+str(base_sigma)+'][num_intervals_per_octave:'+str(num_intervals_per_octave)+'][low_contrast_threshold:'+str(low_contrast_threshold)+']')
                        makedirs(out_dir)
                        sub_html_file_name = os.path.join(out_dir,'sub_link.html')
                        sift_supplemental_html_code += '<BR> <BR> <a href="'+sub_html_file_name+'">[image:'+str(image)+'][num_octaves:'+str(num_octaves)+'][num_octaves:'+str(num_octaves)+'][base_sigma:'+str(base_sigma)+'][num_intervals_per_octave:'+str(num_intervals_per_octave)+'][low_contrast_threshold:'+str(low_contrast_threshold)+']</a>'
                        with open(sub_html_file_name,'wt') as f:
                            f.write('''
<h3> SIFT </h3> <BR>
<hr noshade color="#000000">
<h4> Number Of Octaves: '''+str(num_octaves)+''' </h4>
<h4> Base Sigma: '''+str(base_sigma)+''' </h4>
<h4> Number Of Intervals Per Octave: '''+str(num_intervals_per_octave)+''' </h4>
<h4> Low Contrast Threshold: '''+str(low_contrast_threshold)+''' </h4>

<BR><BR><BR><BR>
<h4> Input Image: </h4> <img src="input.png"> <BR> <BR> <BR>
<h4> SIFT Points: </h4> <img src="out.png"> <BR> <BR> <BR>
''')
                        image_link_name = os.path.join(out_dir,'out.png')
                        if not os.path.isfile(image_link_name):
                            commands += ('python sift.py '+str(image)+' '+str(num_octaves)+' '+str(base_sigma)+' '+str(num_intervals_per_octave)+' '+str(low_contrast_threshold)+' '+str(out_dir)+' & ')
                            job_count+= 1
                        if job_count == 4:
                            job_count = 0
                            system('('+commands+'wait'+') > /dev/null 2>&1')
                            print 
                            commands = '' 
    system('('+commands+'wait'+') > /dev/null 2>&1')
    with open(sift_supplemental_html_file_name,'wt') as f:
        f.write(sift_supplemental_html_code)
    # SIFT Gen End #################################################################################################
    
    html_code += '''
<hr noshade color="#000000">
'''
    write()
    print 'HTML Generation complete!'

if __name__ == '__main__':
    main()


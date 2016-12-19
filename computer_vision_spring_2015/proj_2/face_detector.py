#!/usr/bin/python

# Face Detector

# Standard Libraries
import sys
import os
import pdb
import time
import math
import numpy
import Image
import random
import matplotlib
import matplotlib.pyplot
import scipy.ndimage.filters
import shutil
import warnings

# Non-Standard Libraries
from util import *

PATCH_WIDTH = 12
NUM_PATCHES_PER_SET = 100 # the sets are faces and non-faces
FACE_PADDING = 8
LEARNING_RATE = 1e-6
EPSILON = 1e-6
NON_MAXIMUM_SUPRESSION_WIDTH = 30
BASE_SIGMA = 5
NUM_SCALES = 8
KERNEL_DIM = 5
BLUE = numpy.array([0,0,255], dtype='uint8')
NUM_FINAL_TEST_IMAGES = 5
LOGISTIC_REGRESSION_ITERATION_LIMIT = 1e5

def usage():
    print >> sys.stderr, 'python '+__file__+' <options>'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options: -convert_to_png'
    print >> sys.stderr, '         -training_images <trainings_images_directory_location>'
    print >> sys.stderr, '         -testing_images <testing_images_directory_location>'
    print >> sys.stderr, '         -patch_locations_text_file <text_file_containing_patch_locations>'
    print >> sys.stderr, '         -use_gaussian_model_single_scale'
    print >> sys.stderr, '         -start_clean'
    print >> sys.stderr, '         -use_logistic_regression_model_single_scale'
    print >> sys.stderr, '         -use_gaussian_model_multi_scale'
    print >> sys.stderr, '         -logistic_regression_test'
    sys.exit(1)

def assertion(condition, message):
    if not condition:
        print >> sys.stderr, ''
        print >> sys.stderr, message
        print >> sys.stderr, ''
        usage()
        sys.exit(1)

def get_command_line_param_val(args, param_option, param_option_not_specified_error_message, param_val_not_specified_error_message):
    assertion(param_option in args, param_option_not_specified_error_message)
    param_val_index = 1+args.index(param_option)
    assertion(param_val_index < len(args), param_val_not_specified_error_message)
    return args[param_val_index]

def get_testing_images_directory(args):
    return get_command_line_param_val(args, '-testing_images', 'Error: A testing images directory must be specified.', 'Error: Problem with testing images directory location.')

def get_training_images_directory(args):
    return get_command_line_param_val(args, '-training_images', 'Error: A training images directory must be specified.', 'Error: Problem with training images directory location.')

def get_patch_locations_text_file(args):
    return get_command_line_param_val(args, '-patch_locations_text_file', 'Error: A patch locations text file must be specified.', 'Error: Problem with patch locations text file location.')

def convert_to_png(args):
    # Convert to PNGs via ImageMaick (since Python Image Library is having trouble with opening the GIF images directly)
    input_dir_location_original = get_training_images_directory(args)
    
    gif_dir = os.path.abspath(input_dir_location_original)
    png_dir = os.path.abspath(os.path.join(get_containing_folder(gif_dir), 'pngs'))
    makedirs(png_dir)
    
    count = 0
    commands = ''
    for sub_dir in list_dir_abs(gif_dir):
        png_sub_dir = os.path.join(png_dir, path_leaf(sub_dir))
        makedirs(png_sub_dir)
        for image_location in [ sub_dir_file for sub_dir_file in list_dir_abs(sub_dir) if '.gif' in sub_dir_file[-5:]]:
            commands += 'convert -auto-gamma '+image_location+' '+os.path.join(png_sub_dir,path_leaf(image_location).replace('.gif','.png'))+' & '
            count += 1
            if count == 32:
                commands += 'wait'
                system(commands)
                commands = ''
                count = 0

squared_difference = lambda a,b: (a-b)**2
squared_difference_vectorized = numpy.vectorize(squared_difference)
def patch_ssd(patch_a, patch_b):
    return numpy.sum(squared_difference_vectorized(patch_apatch_b),axis=None)

def gather_patches(args):
    patch_locations = get_patch_locations_text_file(args)
    input_dir_location_original = get_training_images_directory(args)
    gif_dir = os.path.abspath(input_dir_location_original)
    png_dir = os.path.abspath(os.path.join(get_containing_folder(gif_dir), 'pngs'))
    
    lines = sorted(open(patch_locations, 'rt').readlines())
    patch_data = []
    for line in lines:
        if '.gif ' not in line:
            continue
        pieces = line.split()
        patch_data.append({ 'image_name'               :pieces[0].replace('.gif','.png') ,
                            'left_eye_x'               : int(float(pieces[1])) ,
                            'left_eye_y'               : int(float(pieces[2])) ,
                            'right_eye_x'              : int(float(pieces[3])) ,
                            'right_eye_y'              : int(float(pieces[4])) ,
                            'nose_x'                   : int(float(pieces[5])) ,
                            'nose_y'                   : int(float(pieces[6])) ,
                            'left_corner_mouth_x'      : int(float(pieces[7])) ,
                            'left_corner_mouth_y'      : int(float(pieces[8])) ,
                            'center_mouth_x'           : int(float(pieces[9])) ,
                            'center_mouth_y'           : int(float(pieces[10])) ,
                            'right_corner_mouth_x'     : int(float(pieces[11])) ,
                            'right_corner_mouth_y'     : int(float(pieces[12])) })
    if not os.path.isfile('face_patches.py') or not os.path.isfile('non_face_patches.py'):
        face_patches = []
        non_face_patches = []
        for sub_dir in list_dir_abs(png_dir):
            for image_abs_location in [ sub_dir_file for sub_dir_file in list_dir_abs(sub_dir) if '.png' in sub_dir_file[-5:]]:
                patch_coords = []
                image_file_name = path_leaf(image_abs_location)
                image = numpy.asarray(Image.open(image_abs_location))
                h, w = image.shape
                for patch_datum in patch_data:
                    if patch_datum['image_name'] == image_file_name:
                        left_eye_x = patch_datum['left_eye_x']
                        left_eye_y = patch_datum['left_eye_y']
                        right_eye_x = patch_datum['right_eye_x']
                        right_eye_y = patch_datum['right_eye_y']
                        nose_x = patch_datum['nose_x']
                        nose_y = patch_datum['nose_y']
                        left_corner_mouth_x = patch_datum['left_corner_mouth_x']
                        left_corner_mouth_y = patch_datum['left_corner_mouth_y']
                        center_mouth_x = patch_datum['center_mouth_x']
                        center_mouth_y = patch_datum['center_mouth_y']
                        right_corner_mouth_x = patch_datum['right_corner_mouth_x']
                        right_corner_mouth_y = patch_datum['right_corner_mouth_y']
                        x_coords = [left_eye_x, right_eye_x, nose_x, left_corner_mouth_x, center_mouth_x, right_corner_mouth_x]
                        y_coords = [left_eye_y, right_eye_y, nose_y, left_corner_mouth_y, center_mouth_y, right_corner_mouth_y]
                        x_min = min(x_coords)-FACE_PADDING
                        x_max = max(x_coords)+FACE_PADDING
                        y_min = min(y_coords)-FACE_PADDING
                        y_max = max(y_coords)+FACE_PADDING
                        patch0 = image[y_min:y_max,x_min:x_max]
                        patch = numpy.empty([12,12],dtype=image.dtype)
                        scipy.ndimage.interpolation.zoom(patch0, [12.0/(y_max-y_min),12.0/(x_max-x_min)], patch) # use the whole face
#                        patch = image[left_eye_y:left_eye_y+12,left_eye_x:left_eye_x+12] # Use Just the Eye
                        assert patch.shape == (12,12)
                        face_patches.append((patch, image_abs_location))
                        patch_coords.append( (y_max,y_min,x_max,x_min) )
                non_face_patches_to_be_append = []
                non_face_patches_to_be_append_coords = []
                while len(non_face_patches_to_be_append) < len(patch_coords):
                    new_y = random.randint(0,h-PATCH_WIDTH-1)
                    new_x = random.randint(0,w-PATCH_WIDTH-1)
                    if reduce(lambda a,b:a and b, [ (new_x>x_max or new_x+12<x_min or new_y<y_min or new_y+12>y_max) for (y_max,y_min,x_max,x_min) in patch_coords+non_face_patches_to_be_append_coords]):
                        non_face_patches_to_be_append_coords.append( (new_y+12,new_y,new_x,new_x+12) )
                        non_face_patch = image[new_y:new_y+PATCH_WIDTH,new_x:new_x+PATCH_WIDTH]
                        non_face_patches_to_be_append.append(non_face_patch)
                non_face_patches += non_face_patches_to_be_append
                if len(non_face_patches) == 100 and len(patches) == 100 and USE_FIRST_SEQUENCE_OF_PATCHES:
                    break; 
            if len(non_face_patches) == 100 and len(patches) == 100 and USE_FIRST_SEQUENCE_OF_PATCHES:
                break; 
        
        # Save Patches as Collage (Assuming we are using 100 face patches and 100 non-face patches)
        face_patches = random.sample(face_patches, 100)
        training_images = list(set([f[1] for f in face_patches]))
        face_patches = [f[0] for f in face_patches]
        current_training_images = os.path.abspath('./patch_source_images')
        os.system('rm -rf '+current_training_images+' > /dev/null 2>&1')
        makedirs_recursive(current_training_images)
        [shutil.copy(f, current_training_images) for f in training_images]
        non_face_patches = random.sample(non_face_patches, 100)
        with open('face_patches.py', 'wt') as f:
            f.write('import numpy\n\nface_patches = '+repr(face_patches).replace('array','numpy.array').replace('uint8','\'uint8\''))
        with open('non_face_patches.py', 'wt') as f:
            f.write('import numpy\n\nnon_face_patches = '+repr(non_face_patches).replace('array','numpy.array').replace('uint8','\'uint8\''))
        
    l = dict()
    g = dict()
    exec open('face_patches.py', 'rt').read() in l,g
    exec open('non_face_patches.py', 'rt').read() in l,g
    non_face_patches = g['non_face_patches'] 
    face_patches = g['face_patches'] 
    
    collage = numpy.empty([PATCH_WIDTH*10,PATCH_WIDTH*20], dtype='uint8')
    yy, xx = 0, 0
    for i, patch in enumerate(face_patches):
        collage[yy:yy+PATCH_WIDTH,xx:xx+PATCH_WIDTH] = patch
        yy += PATCH_WIDTH
        if yy == PATCH_WIDTH*10:
            yy = 0
            xx += PATCH_WIDTH
    
    yy = 0
    for i, patch in enumerate(non_face_patches):
        collage[yy:yy+PATCH_WIDTH,xx:xx+PATCH_WIDTH] = patch
        yy += PATCH_WIDTH
        if yy == PATCH_WIDTH*10:
            yy = 0
            xx += PATCH_WIDTH
    collage_final = numpy.empty([240,480],dtype='uint8')
    scipy.ndimage.interpolation.zoom(collage, 2.0, collage_final)
    Image.fromarray(collage_final).save('collage.png')
    
    os.system('clear')
    
    return face_patches, non_face_patches

def g(x_i, w):
    ans = 1/(1+numpy.exp(-numpy.dot(w.T,x_i)))
    return ans

def create_logistic_regression_model(face_patches, non_face_patches):
    training_set = [(e,1) for e in face_patches]+[(e,0) for e in non_face_patches]
    w = numpy.zeros(145,dtype='float')
    count = 0
    w0 = numpy.array(w)
    w_change = 0
    while count < 100 or (w_change>EPSILON):
        w_change = abs(numpy.sum(w-w0,axis=None))
        w0 = numpy.array(w)
        summation = 0
        for x_i0,y_i in training_set:
            x_i = numpy.append(x_i0,numpy.array([1.0],dtype='float'))
            summation += ((y_i-g(x_i,w))*x_i)
        w = w + LEARNING_RATE * summation 
        count += 1
        if count > LOGISTIC_REGRESSION_ITERATION_LIMIT: 
            break
    print str(count)+" iterations before convergence."
    print 
    return w
    
def use_logistic_regression_model_single_scale(args):
    print 
    current_output_dir = os.path.abspath('./output')
    os.system('rm -rf '+current_output_dir+' > /dev/null 2>&1')
    makedirs_recursive(current_output_dir)
    num_valid_images = 0
    num_face_images = 0
    face_patches, non_face_patches = gather_patches(args)
    testing_images = os.path.abspath(get_testing_images_directory(args))
    w = create_logistic_regression_model(face_patches, non_face_patches)
    for test_image_file_name in sorted([f for f in list_dir_abs(testing_images) if '.png' in f[-5:]])[:NUM_FINAL_TEST_IMAGES]:
        test_image = numpy.array(Image.open(test_image_file_name))
        if len(test_image.shape) > 2:
            continue
        num_valid_images += 1
        height,width = test_image.shape
        face_probability_map = numpy.zeros([height-PATCH_WIDTH,width-PATCH_WIDTH],dtype='float')
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                patch = test_image[patch_y:patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH]
                x_i = numpy.append(patch.flatten(),numpy.array([1.0],dtype='float'))
                face_probability = g(x_i, w)
                if face_probability > 0.5:
                    face_probability_map[patch_y,patch_x] = face_probability
        if numpy.sum(face_probability_map,axis=None)>0:
            num_face_images += 1
            print test_image_file_name+" contains a face"
        # Non-Maximum Suppression
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                max_value = numpy.max(face_probability_map[patch_y:patch_y+NON_MAXIMUM_SUPRESSION_WIDTH,patch_x:patch_x+NON_MAXIMUM_SUPRESSION_WIDTH],axis=None)
                max_found = False
                for yy in xrange(patch_y,min(height-PATCH_WIDTH,patch_y+NON_MAXIMUM_SUPRESSION_WIDTH)):
                    for xx in xrange(patch_x,min(width-PATCH_WIDTH,patch_x+NON_MAXIMUM_SUPRESSION_WIDTH)):
                        if max_found or face_probability_map[yy,xx] < max_value:
                            face_probability_map[yy,xx] = 0
                        else:
                            max_found = True
        out_image = numpy.empty([height,width,3],dtype='uint8')
        out_image[:,:,0] = test_image
        out_image[:,:,1] = test_image
        out_image[:,:,2] = test_image
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                if face_probability_map[patch_y,patch_x] > 0:
                    out_image[patch_y:patch_y+PATCH_WIDTH,patch_x] = [0,0,255]
                    out_image[patch_y:patch_y+PATCH_WIDTH,patch_x+PATCH_WIDTH] = [0,0,255]
                    out_image[patch_y,patch_x:patch_x+PATCH_WIDTH] = [0,0,255]
                    out_image[patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH] = [0,0,255]
        Image.fromarray(out_image).save(os.path.join(current_output_dir,path_leaf(test_image_file_name)))
    print 
    print 'Num Face Images: '+str(num_face_images)+'/'+str(num_valid_images)
    print 

def use_logistic_regression_model_multi_scale(args):
    print 
    current_output_dir = os.path.abspath('./output')
    os.system('rm -rf '+current_output_dir+' > /dev/null 2>&1')
    makedirs_recursive(current_output_dir)
    num_valid_images = 0
    num_face_images = 0
    face_patches, non_face_patches = gather_patches(args)
    testing_images = os.path.abspath(get_testing_images_directory(args))
    w = create_logistic_regression_model(face_patches, non_face_patches)
    gaussian_kernel = get_gaussian_kernel(KERNEL_DIM, BASE_SIGMA)
    for test_image_file_name in sorted([f for f in list_dir_abs(testing_images) if '.png' in f[-5:]])[:NUM_FINAL_TEST_IMAGES]:
        test_image = numpy.array(Image.open(test_image_file_name))
        if len(test_image.shape) > 2:
            continue
        height,width = test_image.shape
        out_image = numpy.empty([height,width,3],dtype='uint8')
        out_image[:,:,0] = test_image
        out_image[:,:,1] = test_image
        out_image[:,:,2] = test_image
        num_valid_images += 1
        face_found = False
        face_probability_map_list = []
        for scale in xrange(NUM_SCALES):
            height,width = test_image.shape
            if height-PATCH_WIDTH<0 or width-PATCH_WIDTH<0:
                break
            face_probability_map = numpy.zeros([height-PATCH_WIDTH,width-PATCH_WIDTH],dtype='float')
            for patch_y in xrange(height-PATCH_WIDTH):
                for patch_x in xrange(width-PATCH_WIDTH):
                    patch = test_image[patch_y:patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH]
                    x_i = numpy.append(patch.flatten(),numpy.array([1.0],dtype='float'))
                    face_probability = g(x_i, w)
                    if face_probability > 0.5:
                        face_probability_map[patch_y,patch_x] = face_probability
            face_found = face_found or numpy.sum(face_probability_map,axis=None)>0
            # Non-Maximum Suppression
            for patch_y in xrange(height-PATCH_WIDTH):
                for patch_x in xrange(width-PATCH_WIDTH):
                    max_value = numpy.max(face_probability_map[patch_y:patch_y+NON_MAXIMUM_SUPRESSION_WIDTH,patch_x:patch_x+NON_MAXIMUM_SUPRESSION_WIDTH],axis=None)
                    max_found = False
                    for yy in xrange(patch_y,min(height-PATCH_WIDTH,patch_y+NON_MAXIMUM_SUPRESSION_WIDTH)):
                        for xx in xrange(patch_x,min(width-PATCH_WIDTH,patch_x+NON_MAXIMUM_SUPRESSION_WIDTH)):
                            if max_found or face_probability_map[yy,xx] < max_value:
                                face_probability_map[yy,xx] = 0
                            else:
                                max_found = True
            test_image = downsample_2d(test_image,2)
            test_image = convolve(test_image, gaussian_kernel)
            face_probability_map_list.append(face_probability_map)
        out_image_height, out_image_width = out_image.shape[:2]
        claimed_spots_map = numpy.ones(out_image.shape,dtype='uint8')
        for scale_index in reversed(xrange(len(face_probability_map_list))):
            face_probability_map = face_probability_map_list[scale_index]
            face_probability_map_height, face_probability_map_width = face_probability_map.shape[:2]
            for patch_y in xrange(face_probability_map_height-PATCH_WIDTH):
                for patch_x in xrange(face_probability_map_width-PATCH_WIDTH):
                    if face_probability_map[patch_y,patch_x] > 0:
                        yy = patch_y*(1+scale_index)
                        xx = patch_x*(1+scale_index)
                        scaled_patch_width = PATCH_WIDTH*(1+scale_index)
                        if not numpy.any(claimed_spots_map[yy:yy+scaled_patch_width,xx:xx+scaled_patch_width]>5):
                            claimed_spots_map[yy:yy+scaled_patch_width,xx:xx+scaled_patch_width,:] *= 255
                            out_image[yy:yy+scaled_patch_width,xx] = BLUE
                            out_image[yy:yy+scaled_patch_width,xx+scaled_patch_width] = BLUE
                            out_image[yy,xx:xx+scaled_patch_width] = BLUE
                            out_image[yy+scaled_patch_width,xx:xx+scaled_patch_width] = BLUE
        if face_found:
            num_face_images += 1
            Image.fromarray(out_image).save(os.path.join(current_output_dir,path_leaf(test_image_file_name)))
#            Image.fromarray(claimed_spots_map).save(os.path.join(current_output_dir,"map-"+path_leaf(test_image_file_name)))
            print test_image_file_name+" contains a face"
    print 
    print 'Num Face Images: '+str(num_face_images)+'/'+str(num_valid_images)
    print 
    

def create_gaussian_model(face_patches, non_face_patches):
    
    face_mean_patch = numpy.zeros([PATCH_WIDTH,PATCH_WIDTH],dtype='float')
    non_face_mean_patch = numpy.zeros([PATCH_WIDTH,PATCH_WIDTH],dtype='float')
    for i in xrange(100):
        face_mean_patch += face_patches[i].astype('float')
        non_face_mean_patch += non_face_patches[i].astype('float')
    
    face_mean_patch /= 100.0
    save_image(face_mean_patch, 'face_mean_patch.png')
    non_face_mean_patch /= 100.0
    save_image(non_face_mean_patch, 'non_face_mean_patch.png')
    
    face_mean_patch_big = numpy.empty([150,150],dtype=face_mean_patch.dtype)
    scipy.ndimage.interpolation.zoom(face_mean_patch, 150.0/12.0, face_mean_patch_big)
    save_image(face_mean_patch_big, 'face_mean_patch_big.png')
    
    non_face_mean_patch_big = numpy.empty([150,150],dtype=non_face_mean_patch.dtype)
    scipy.ndimage.interpolation.zoom(non_face_mean_patch, 150.0/12.0, non_face_mean_patch_big)
    save_image(non_face_mean_patch_big, 'non_face_mean_patch_big.png')
    
    face_vectors = numpy.vstack(map(lambda e:e.flatten(),face_patches))
    non_face_vectors = numpy.vstack(map(lambda e:e.flatten(),non_face_patches))
    
    assert non_face_vectors.shape == (100,144)
    assert face_vectors.shape == (100,144)
    
    face_covariance_matrix = numpy.cov(face_vectors.T)
    non_face_covariance_matrix = numpy.cov(non_face_vectors.T)
    
    face_u, face_s, face_v = numpy.linalg.svd(face_covariance_matrix)
    non_face_u, non_face_s, non_face_v = numpy.linalg.svd(non_face_covariance_matrix)
    
    face_fig, face_subplot = matplotlib.pyplot.subplots()
    non_face_fig, non_face_subplot = matplotlib.pyplot.subplots()
    
    face_subplot.scatter(range(len(face_s)),face_s)
    non_face_subplot.scatter(range(len(non_face_s)),non_face_s)
    
    face_subplot.set_title('Singular Values for Face Patches')
    face_subplot.set_ylabel('Singular Value')
    face_subplot.set_xlabel('Patch Index')
    face_subplot.set_ylim(bottom=0)
#    face_subplot.set_ylim(top=10)
#    face_subplot.set_yscale('log')
    face_subplot.set_xlim(left=0)
    face_subplot.set_xlim(right=144)
    
    non_face_subplot.set_title('Singular Values for Non-Face Patches')
    non_face_subplot.set_ylabel('Singular Value')
    non_face_subplot.set_xlabel('Patch Index')
    non_face_subplot.set_ylim(bottom=0)
#    non_face_subplot.set_ylim(top=10)
#    non_face_subplot.set_yscale('log')
    non_face_subplot.set_xlim(left=0)
    non_face_subplot.set_xlim(right=144)
    
    face_fig.savefig('face_singular_values.png')
    non_face_fig.savefig('non_face_singular_values.png')
    
    tau = 3000
    
    face_s_thresholded = [e for e in face_s if e>tau]
    non_face_s_thresholded = [e for e in non_face_s if e>tau]
    
    face_fig_thresholded, face_subplot_thresholded = matplotlib.pyplot.subplots()
    non_face_fig_thresholded, non_face_subplot_thresholded = matplotlib.pyplot.subplots()

    face_subplot_thresholded.scatter(range(len(face_s_thresholded)),face_s_thresholded)
    non_face_subplot_thresholded.scatter(range(len(non_face_s_thresholded)),non_face_s_thresholded)

    face_subplot_thresholded.set_title('Singular Values for Face Patches')
    face_subplot_thresholded.set_ylabel('Singular Value')
    face_subplot_thresholded.set_xlabel('Patch Index')
    face_subplot_thresholded.set_ylim(bottom=0)
    face_subplot_thresholded.set_xlim(left=0)

    non_face_subplot_thresholded.set_title('Singular Values for Non-Face Patches')
    non_face_subplot_thresholded.set_ylabel('Singular Value')
    non_face_subplot_thresholded.set_xlabel('Patch Index')
    non_face_subplot_thresholded.set_ylim(bottom=0)
    non_face_subplot_thresholded.set_xlim(left=0)
    
    face_subplot_thresholded.set_xlim(right=20)
    non_face_subplot_thresholded.set_xlim(right=20)
    face_fig_thresholded.savefig('face_singular_values_thresholded.png')
    non_face_fig_thresholded.savefig('non_face_singular_values_thresholded.png')
    
    assert numpy.ndarray.tolist(face_s)==sorted(face_s, reverse=True)
    assert numpy.ndarray.tolist(non_face_s)==sorted(non_face_s, reverse=True)
    
    face_K = len(face_s_thresholded)
    non_face_K = len(non_face_s_thresholded)
    
    face_Sk = numpy.zeros([face_K,face_K],dtype=face_s.dtype)
    for i in xrange(face_K):
        face_Sk[i,i] = face_s[i]
    non_face_Sk = numpy.zeros([non_face_K,non_face_K],dtype=non_face_s.dtype)
    for i in xrange(non_face_K):
        non_face_Sk[i,i] = non_face_s[i]
    
    face_Uk = face_u[:face_K,:face_K]
    non_face_Uk = non_face_u[:non_face_K,:non_face_K]
    
    face_Ek = numpy.dot(numpy.dot(face_Uk,face_Sk),face_Uk.T)
    non_face_Ek = numpy.dot(numpy.dot(non_face_Uk,non_face_Sk),non_face_Uk.T)
    
    face_Ek_determinant = numpy.linalg.det(face_Ek)
    non_face_Ek_determinant = numpy.linalg.det(non_face_Ek)
    
    face_mean_vector = face_mean_patch.flatten()
    non_face_mean_vector = non_face_mean_patch.flatten()
    
    model = locals()
    
    print "face_K:", face_K 
    print "non_face_K:", non_face_K
    
    return model

def get_probability_via_gaussian_model(input_vector0, mean_vector0, covariance_matrix, covariance_matrix_determinant, model):
    k = covariance_matrix.shape[0]
    input_vector = input_vector0[:k]
    mean_vector = mean_vector0[:k]
    difference_from_mean = input_vector-mean_vector
    return (1.0/math.sqrt(covariance_matrix_determinant*(math.pi*2)**k))*math.exp(-0.5*numpy.dot(numpy.dot((difference_from_mean).T,numpy.linalg.inv(covariance_matrix)),difference_from_mean))

def get_probability_is_face_via_gaussian_model(input_patch, model):
    input_vector = input_patch.flatten()
    face_mean_vector = model['face_mean_vector'] 
    face_Ek = model['face_Ek'] 
    face_Ek_determinant = model['face_Ek_determinant'] 
    return get_probability_via_gaussian_model(input_vector, face_mean_vector, face_Ek, face_Ek_determinant, model)

def get_probability_is_non_face_via_gaussian_model(input_patch, model):
    input_vector = input_patch.flatten()
    non_face_mean_vector = model['non_face_mean_vector'] 
    non_face_Ek = model['non_face_Ek'] 
    non_face_Ek_determinant = model['non_face_Ek_determinant'] 
    return get_probability_via_gaussian_model(input_vector, non_face_mean_vector, non_face_Ek, non_face_Ek_determinant, model)

def use_gaussian_model_single_scale(args):
    print 
    current_output_dir = os.path.abspath('./output')
    os.system('rm -rf '+current_output_dir+' > /dev/null 2>&1')
    makedirs_recursive(current_output_dir)
    num_valid_images = 0
    num_face_images = 0
    face_patches, non_face_patches = gather_patches(args)
    testing_images = os.path.abspath(get_testing_images_directory(args))
    gaussian_model = create_gaussian_model(face_patches, non_face_patches)
    for test_image_file_name in sorted([f for f in list_dir_abs(testing_images) if '.png' in f[-5:]])[:NUM_FINAL_TEST_IMAGES]:
        test_image = numpy.array(Image.open(test_image_file_name))
        if len(test_image.shape) > 2:
            continue
        num_valid_images += 1
        height,width = test_image.shape
        face_probability_map = numpy.zeros([height-PATCH_WIDTH,width-PATCH_WIDTH],dtype='float')
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                patch = test_image[patch_y:patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH]
                probability_difference = get_probability_is_face_via_gaussian_model(patch, gaussian_model) - get_probability_is_non_face_via_gaussian_model(patch, gaussian_model)
                if probability_difference > 0:
                    face_probability_map[patch_y,patch_x] = probability_difference
        if numpy.sum(face_probability_map,axis=None)>0:
            num_face_images += 1
            print test_image_file_name+" contains a face"
        # Non-Maximum Suppression
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                max_value = numpy.max(face_probability_map[patch_y:patch_y+NON_MAXIMUM_SUPRESSION_WIDTH,patch_x:patch_x+NON_MAXIMUM_SUPRESSION_WIDTH],axis=None)
                max_found = False
                for yy in xrange(patch_y,min(height-PATCH_WIDTH,patch_y+NON_MAXIMUM_SUPRESSION_WIDTH)):
                    for xx in xrange(patch_x,min(width-PATCH_WIDTH,patch_x+NON_MAXIMUM_SUPRESSION_WIDTH)):
                        if max_found or face_probability_map[yy,xx] < max_value:
                            face_probability_map[yy,xx] = 0
                        else:
                            max_found = True
        out_image = numpy.empty([height,width,3],dtype='uint8')
        out_image[:,:,0] = test_image
        out_image[:,:,1] = test_image
        out_image[:,:,2] = test_image
        for patch_y in xrange(height-PATCH_WIDTH):
            for patch_x in xrange(width-PATCH_WIDTH):
                if face_probability_map[patch_y,patch_x] > 0:
                    out_image[patch_y:patch_y+PATCH_WIDTH,patch_x] = [0,0,255]
                    out_image[patch_y:patch_y+PATCH_WIDTH,patch_x+PATCH_WIDTH] = [0,0,255]
                    out_image[patch_y,patch_x:patch_x+PATCH_WIDTH] = [0,0,255]
                    out_image[patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH] = [0,0,255]
        Image.fromarray(out_image).save(os.path.join(current_output_dir,path_leaf(test_image_file_name)))
    print 
    print 'Num Face Images: '+str(num_face_images)+'/'+str(num_valid_images)
    print 

def use_gaussian_model_multi_scale(args):
    print 
    current_output_dir = os.path.abspath('./output')
    os.system('rm -rf '+current_output_dir+' > /dev/null 2>&1')
    makedirs_recursive(current_output_dir)
    num_valid_images = 0
    num_face_images = 0
    face_patches, non_face_patches = gather_patches(args)
    testing_images = os.path.abspath(get_testing_images_directory(args))
    gaussian_model = create_gaussian_model(face_patches, non_face_patches)
    gaussian_kernel = get_gaussian_kernel(KERNEL_DIM, BASE_SIGMA)
    for test_image_file_name in sorted([f for f in list_dir_abs(testing_images) if '.png' in f[-5:]])[:NUM_FINAL_TEST_IMAGES]:
        test_image = numpy.array(Image.open(test_image_file_name))
        if len(test_image.shape) > 2:
            continue
        out_image = numpy.empty(list(test_image.shape)+[3],dtype='uint8')
        out_image[:,:,0] = test_image
        out_image[:,:,1] = test_image
        out_image[:,:,2] = test_image
        num_valid_images += 1
        face_found = False
        face_probability_map_list = []
        for scale in xrange(NUM_SCALES):
            height,width = test_image.shape
            if height-PATCH_WIDTH<0 or width-PATCH_WIDTH<0:
                break
            face_probability_map = numpy.zeros([height-PATCH_WIDTH,width-PATCH_WIDTH],dtype='float')
            for patch_y in xrange(height-PATCH_WIDTH):
                for patch_x in xrange(width-PATCH_WIDTH):
                    patch = test_image[patch_y:patch_y+PATCH_WIDTH,patch_x:patch_x+PATCH_WIDTH]
                    probability_difference = get_probability_is_face_via_gaussian_model(patch, gaussian_model) - get_probability_is_non_face_via_gaussian_model(patch, gaussian_model)
                    if probability_difference > 0:
                        face_probability_map[patch_y,patch_x] = probability_difference
            face_found = face_found or numpy.sum(face_probability_map,axis=None)>0
            # Non-Maximum Suppression
            for patch_y in xrange(height-PATCH_WIDTH):
                for patch_x in xrange(width-PATCH_WIDTH):
                    max_value = numpy.max(face_probability_map[patch_y:patch_y+NON_MAXIMUM_SUPRESSION_WIDTH,patch_x:patch_x+NON_MAXIMUM_SUPRESSION_WIDTH],axis=None)
                    max_found = False
                    for yy in xrange(patch_y,min(height-PATCH_WIDTH,patch_y+NON_MAXIMUM_SUPRESSION_WIDTH)):
                        for xx in xrange(patch_x,min(width-PATCH_WIDTH,patch_x+NON_MAXIMUM_SUPRESSION_WIDTH)):
                            if max_found or face_probability_map[yy,xx] < max_value:
                                face_probability_map[yy,xx] = 0
                            else:
                                max_found = True
            test_image = downsample_2d(test_image,2)
            test_image = convolve(test_image, gaussian_kernel)
            face_probability_map_list.append(face_probability_map)
        out_image_height, out_image_width = out_image.shape[:2]
        claimed_spots_map = numpy.ones(out_image.shape,dtype='uint8')
        for scale_index in reversed(xrange(len(face_probability_map_list))):
            face_probability_map = face_probability_map_list[scale_index]
            face_probability_map_height, face_probability_map_width = face_probability_map.shape[:2]
            for patch_y in xrange(face_probability_map_height-PATCH_WIDTH):
                for patch_x in xrange(face_probability_map_width-PATCH_WIDTH):
                    if face_probability_map[patch_y,patch_x] > 0:
                        yy = patch_y*(1+scale_index)
                        xx = patch_x*(1+scale_index)
                        scaled_patch_width = PATCH_WIDTH*(1+scale_index)
                        if not numpy.any(claimed_spots_map[yy:yy+scaled_patch_width,xx:xx+scaled_patch_width]>5):
                            claimed_spots_map[yy:yy+scaled_patch_width,xx:xx+scaled_patch_width,:] *= 255
                            out_image[yy:yy+scaled_patch_width,xx] = BLUE
                            out_image[yy:yy+scaled_patch_width,xx+scaled_patch_width] = BLUE
                            out_image[yy,xx:xx+scaled_patch_width] = BLUE
                            out_image[yy+scaled_patch_width,xx:xx+scaled_patch_width] = BLUE
        if face_found:
            num_face_images += 1
            Image.fromarray(out_image).save(os.path.join(current_output_dir,path_leaf(test_image_file_name)))
#            Image.fromarray(claimed_spots_map).save(os.path.join(current_output_dir,"map-"+path_leaf(test_image_file_name)))
            print test_image_file_name+" contains a face"
    print 
    print 'Num Face Images: '+str(num_face_images)+'/'+str(num_valid_images)
    print 

def main():
    # Sample Usage: ./face_detector.py -gather_patches -training_images ./gifs/ -patch_locations_text_file patch_locations_a.txt -use_gaussian_model_single_scale -testing_images ./test_faces -start_clean
    # Sample Usage: ./face_detector.py -gather_patches -training_images ./gifs/ -patch_locations_text_file patch_locations_a.txt -use_logistic_regression_model_single_scale -testing_images ./test_faces -start_clean
    # Sample Usage: ./face_detector.py -gather_patches -training_images ./gifs/ -patch_locations_text_file patch_locations_a.txt -use_gaussian_model_multi_scale -testing_images ./test_faces -start_clean
    # Sample Usage: ./face_detector.py -gather_patches -training_images ./gifs/ -patch_locations_text_file patch_locations_a.txt -use_logistic_regression_model_multi_scale -testing_images ./test_faces -start_clean
    print 
    if len(sys.argv) < 2:
        usage()
    
    if '-start_clean' in sys.argv:
       os.system('rm collage.png face_mean_patch_big.png face_mean_patch.png face_singular_values.png face_singular_values_thresholded.png non_face_mean_patch_big.png non_face_mean_patch.png non_face_singular_values.png non_face_singular_values_thresholded.png face_patches.py non_face_patches.py > /dev/null 2>&1') 
    
    if '-logistic_regression_test' in sys.argv:
        face_patches, non_face_patches = gather_patches(sys.argv)
        limits = range(10,6000,10)
        accuracies = []
        for index,limit in enumerate(limits):
            print "Limit:", limit
#            print zip(limits[:index],accuracies)
            global LOGISTIC_REGRESSION_ITERATION_LIMIT
            LOGISTIC_REGRESSION_ITERATION_LIMIT = limit
            w = create_logistic_regression_model(face_patches, non_face_patches)
            num_runs = 0
            num_correct = 0
            for patch in face_patches:
                x_i = numpy.append(patch.flatten(),numpy.array([1.0],dtype='float'))
                face_probability = g(x_i, w)
                num_runs += 1 
                if face_probability > 0.5:
                    num_correct += 1
            for patch in non_face_patches:
                x_i = numpy.append(patch.flatten(),numpy.array([1.0],dtype='float'))
                face_probability = g(x_i, w)
                num_runs += 1 
                if face_probability < 0.5:
                    num_correct += 1
            accuracy = float(num_correct)/float(num_runs)
            accuracies.append(accuracy)
            fig, subplot = matplotlib.pyplot.subplots()
            subplot.set_title('Logistic Regression Accuracy')
            subplot.set_ylabel('Accuracy')
            subplot.set_xlabel('Num Iterations')
            subplot.set_ylim(bottom=0)
            subplot.set_ylim(top=1.0)
#            subplot.set_xlim(left=0)
            subplot.scatter(limits[:index+1],accuracies)
            fig.savefig('logistic_regression_accuracy.png')
#            if len(accuracies)>5:
#                if accuracies[-1]-accuracies[-2]==0:
#                    break
            
        print accuracies
#        pdb.set_trace()
    
    if '-convert_to_png' in sys.argv:
        convert_to_png(sys.argv)
    
    if '-use_gaussian_model_single_scale' in sys.argv:
        use_gaussian_model_single_scale(sys.argv)
    
    if '-use_logistic_regression_model_single_scale' in sys.argv:
        use_logistic_regression_model_single_scale(sys.argv)
    
    if '-use_gaussian_model_multi_scale' in sys.argv:
        use_gaussian_model_multi_scale(sys.argv)
    
    if '-use_logistic_regression_model_multi_scale' in sys.argv:
        use_logistic_regression_model_multi_scale(sys.argv)

if __name__ == '__main__':
    main()


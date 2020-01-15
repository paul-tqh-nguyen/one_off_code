#!/usr/bin/python

import matplotlib
import matplotlib.pyplot
import random

matplotlib.rcParams['lines.linewidth'] = 5

PARTITION_POINTS=20
MAX_DIM=100
NUM_CLUSTERS=50
NUM_CLUSTER_POINTS_MIN=200
NUM_CLUSTER_POINTS_MAX=400

def is_between(x,min_val,max_val):
    return min_val<=x and x<=max_val

def is_below_line(input_x, input_y, start_x, start_y, end_x, end_y):
    m = (float(end_y)-start_y)/(float(end_x)-start_x)
    b = start_y-m*start_x
    return input_y<input_x*m+b

fig, subplot = matplotlib.pyplot.subplots()
subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)
partition_x=[]
partition_y=[]
for i in xrange(PARTITION_POINTS):
    partition_x.append(random.randint(1,MAX_DIM))
    partition_y.append(random.randint(1,MAX_DIM))
partition_y=map(lambda l:(l[0]+l[1])/2.0,zip(partition_x,partition_y))
partition_x.append(0)
partition_y.append(0)
partition_x.append(MAX_DIM)
partition_y.append(MAX_DIM)
partition_x.sort()
partition_y.sort()
subplot.plot(partition_x, partition_y, zorder=10, c=(0,0.85,0), alpha=1.0)
subplot.set_title('SVM Partition of SIFT Features'   )
#subplot.set_xlabel('Inverse N Values')
#subplot.set_ylabel('Inverse P Values')
subplot.set_xlim(left=0, right=MAX_DIM)
subplot.set_ylim(bottom=0, top=MAX_DIM)
for cluster_center_index in xrange(NUM_CLUSTERS):
    x=[]
    y=[]
    num_cluster_points=random.randint(NUM_CLUSTER_POINTS_MIN,NUM_CLUSTER_POINTS_MAX)
    cluster_center_x=random.uniform(1,99)
    cluster_center_y=random.uniform(1,99)
    dot_color = 'r'
    for index in xrange(len(partition_x)-1):
        if is_between(cluster_center_x,partition_x[index],partition_x[index+1]):
            std_dev=random.uniform(3,5)
            if is_below_line(cluster_center_x,cluster_center_y,partition_x[index],partition_y[index],partition_x[index+1],partition_y[index+1]):
                dot_color = 'b'
                break
    for cluster_point_index in xrange(num_cluster_points):
        x.append(random.gauss(cluster_center_x,std_dev))
        y.append(random.gauss(cluster_center_y,std_dev))
    subplot.scatter(x, y, zorder=10, c=dot_color, alpha=0.1)
my_dpi = 100 # See http://www.infobyip.com/detectmonitordpi.php 
width_pixels=800
height_pixels=600
width_inches=width_pixels/float(my_dpi)
height_inches=height_pixels/float(my_dpi)
fig.savefig('./svm.png', figsize=(width_inches, height_inches), dpi=my_dpi)


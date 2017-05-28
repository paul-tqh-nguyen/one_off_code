#!/usr/bin/python

from Mesh import *
import random
import pdb
import time
from math import sin, cos, pi

def cube(length=1):
    m=Mesh()
    m.add_face([
                (0,length,0),
                (length,length,0),
                (length,0,0),
                (0,0,0),
                ]) 
    m.add_face([
                (0,0,length),
                (length,0,length),
                (length,length,length),
                (0,length,length),
                ]) 
    m.add_face([
                (0,0,0),
                (length,0,0),
                (length,0,length),
                (0,0,length),
                ]) 
    m.add_face([
                (0,length,length),
                (length,length,length),
                (length,length,0),
                (0,length,0),
                ]) 
    m.add_face([
                (0,0,length),
                (0,length,length),
                (0,length,0),
                (0,0,0),
                ])
    m.add_face([
                (length,0,0),
                (length,length,0),
                (length,length,length),
                (length,0,length),
                ])
    return m

def cone(height=10, radius=5, num_triangles=360):
    # num_triangles is the number of triangles used for the part of the cone that isn't the base 
    m=Mesh()
    for triangle_index in range(num_triangles):
        start_angle = 2*pi/float(num_triangles)*triangle_index
        end_angle = 2*pi/float(num_triangles)*(triangle_index+1)
        m.add_face([
                    (0,0,height),
                    (radius*sin(end_angle),radius*cos(end_angle),0),
                    (radius*sin(start_angle),radius*cos(start_angle),0),
                    ])
    m.add_face(
        [(radius*sin(2*pi/float(num_triangles)*triangle_index),radius*cos(2*pi/float(num_triangles)*triangle_index),0) for triangle_index in range(num_triangles)]
    )
    for triangle_index in range(num_triangles):
        start_angle = 2*pi/float(num_triangles)*triangle_index
    return m

def torus(inner_radius=5, outer_radius=10, num_segments=36, segment_precision=36):
    # num_segments refers to the number of segments we split the donut/torus into (we cut from the center outward)
    # segment_precision refers to the number of rectangles used per segment 
    m=Mesh()
    assert inner_radius < outer_radius
    tube_radius = (outer_radius-inner_radius)/2.0
    for segment_index in range(num_segments): # index along the length of the tube (the long part if we're thinking about a regular donut)
        lengthwise_start_angle = 2*pi/float(num_segments)*segment_index
        lengthwise_end_angle = 2*pi/float(num_segments)*(segment_index+1)
        lengthwise_tube_start_center_x = (inner_radius+tube_radius)*cos(lengthwise_start_angle)
        lengthwise_tube_start_center_y = (inner_radius+tube_radius)*sin(lengthwise_start_angle)
        lengthwise_tube_start_center_z = 0
        lengthwise_tube_end_center_x = (inner_radius+tube_radius)*cos(lengthwise_end_angle)
        lengthwise_tube_end_center_y = (inner_radius+tube_radius)*sin(lengthwise_end_angle)
        lengthwise_tube_end_center_z = 0
        for rect_index in range(segment_precision): # index along the tube's circumference
            slicewise_tube_start_angle = 2*pi/float(segment_precision)*rect_index
            slicewise_tube_end_angle = 2*pi/float(segment_precision)*(rect_index+1)
            # innertube coordinates
            start_circle_coords = rotate_about_z_axis(lengthwise_start_angle, tube_radius*cos(slicewise_tube_start_angle),0,tube_radius*sin(slicewise_tube_start_angle))
            start_circle_coords_further_along_slice = rotate_about_z_axis(lengthwise_start_angle, tube_radius*cos(slicewise_tube_end_angle),0,tube_radius*sin(slicewise_tube_end_angle))
            end_circle_coords = rotate_about_z_axis(lengthwise_end_angle, tube_radius*cos(slicewise_tube_start_angle),0,tube_radius*sin(slicewise_tube_start_angle))
            end_circle_coords_further_along_slice = rotate_about_z_axis(lengthwise_end_angle, tube_radius*cos(slicewise_tube_end_angle),0,tube_radius*sin(slicewise_tube_end_angle))
            m.add_face([
                        (lengthwise_tube_end_center_x+end_circle_coords[X_COORD],lengthwise_tube_end_center_y+end_circle_coords[Y_COORD],lengthwise_tube_end_center_z+end_circle_coords[Z_COORD]),
                        (lengthwise_tube_end_center_x+end_circle_coords_further_along_slice[X_COORD],lengthwise_tube_end_center_y+end_circle_coords_further_along_slice[Y_COORD],lengthwise_tube_end_center_z+end_circle_coords_further_along_slice[Z_COORD]),
                        (lengthwise_tube_start_center_x+start_circle_coords_further_along_slice[X_COORD],lengthwise_tube_start_center_y+start_circle_coords_further_along_slice[Y_COORD],lengthwise_tube_start_center_z+start_circle_coords_further_along_slice[Z_COORD]),
                        (lengthwise_tube_start_center_x+start_circle_coords[X_COORD],lengthwise_tube_start_center_y+start_circle_coords[Y_COORD],lengthwise_tube_start_center_z+start_circle_coords[Z_COORD]),
                        ])
    return m

def main():
    #m=cube()
    #m=cone()
    m=torus()
    m.save_to_obj_file("C:/Users/nguye/Desktop/out.obj")

if __name__ == '__main__':
    main()
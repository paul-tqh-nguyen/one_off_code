#!/usr/bin/python

from Mesh import Mesh
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
                    
def main():
    #m=cube()
    m=cone()
    m.save_to_obj_file("C:/Users/nguye/Desktop/out.obj")

if __name__ == '__main__':
    main()
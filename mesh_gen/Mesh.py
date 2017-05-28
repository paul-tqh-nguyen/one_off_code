#!/usr/bin/python

import random
import pdb
import time
from math import cos, sin, pi

X_COORD=0
Y_COORD=1
Z_COORD=2

def rotate_about_x_axis(angle_radians, *args):
    if len(args)==1:
        x0 = args[0][X_COORD]
        y0 = args[0][Y_COORD]
        z0 = args[0][Z_COORD]
    elif len(args)==3:
        x0 = args[X_COORD]
        y0 = args[Y_COORD]
        z0 = args[Z_COORD]
    else: 
        assert False
    x = x0 
    y = cos(angle_radians)*y0-sin(angle_radians)*z0
    z = sin(angle_radians)*y0+cos(angle_radians)*z0
    return (x,y,z)

def rotate_about_y_axis(angle_radians, *args):
    if len(args)==1:
        x0 = args[0][X_COORD]
        y0 = args[0][Y_COORD]
        z0 = args[0][Z_COORD]
    elif len(args)==3:
        x0 = args[X_COORD]
        y0 = args[Y_COORD]
        z0 = args[Z_COORD]
    else: 
        assert False
    x = cos(angle_radians)*x0+sin(angle_radians)*z0 
    y = y0
    z = -sin(angle_radians)*x0+cos(angle_radians)*z0
    return (x,y,z)

def rotate_about_z_axis(angle_radians, *args):
    if len(args)==1:
        x0 = args[0][X_COORD]
        y0 = args[0][Y_COORD]
        z0 = args[0][Z_COORD]
    elif len(args)==3:
        x0 = args[X_COORD]
        y0 = args[Y_COORD]
        z0 = args[Z_COORD]
    else: 
        assert False
    x = cos(angle_radians)*x0-sin(angle_radians)*y0
    y = sin(angle_radians)*x0+cos(angle_radians)*y0
    z = z0
    return (x,y,z)

class Mesh(object): 
    
    def __init__(self): 
        self.vertex_dict = dict() # Takes a vertex, returns the index of it. 
        self.face_list = [] # faces are lists. These lists contain vertex indices.
        self.vertex_count=0
    
    def add_vertex(self, *args):
        if len(args)==1:
            x = round(args[0][X_COORD],10)
            y = round(args[0][Y_COORD],10)
            z = round(args[0][Z_COORD],10)
        elif len(args)==3:
            x = round(args[X_COORD],10)
            y = round(args[Y_COORD],10)
            z = round(args[Z_COORD],10)
        else: 
            assert False
        if (x,y,z) not in self.vertex_dict:
            self.vertex_dict[(x,y,z)]= self.vertex_count+1 
            self.vertex_count += 1
        return (x,y,z)
    
    def add_face(self, v_list):
        v_list = map(self.add_vertex, v_list)
        self.face_list.append([self.vertex_dict[v] for v in v_list])
    
    def save_to_obj_file(self, output_file):
        with open(output_file,'w') as f:
            f.write("# Vertices\n")
            for i,(coordinate,index) in enumerate(sorted(self.vertex_dict.items(),key=lambda x:x[1])):
                assert i+1 == index
                f.write("v "+str(coordinate[X_COORD])+" "+str(coordinate[Y_COORD])+" "+str(coordinate[Z_COORD])+"\n") 
            f.write("\n\n")
            f.write("# Faces\n")
            for face in self.face_list:
                f.write("f "+"".join([str(v_index)+" " for v_index in face])+"\n")

def main():
    # Test code
    mesh = Mesh()
    mesh.add_face([(0,0,0),(1,0,0),(0,1,0)])
    mesh.save_to_obj_file("./out.obj")
    

if __name__ == '__main__':
    main()
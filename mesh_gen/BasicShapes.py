#!/usr/bin/python

from Mesh import Mesh
import random
import pdb
import time

def cube(length):
    m=Mesh()
    m.add_face([(0,0,0),
                  (length,0,0),
                  (length,length,0),
                  (0,length,0)]
                    ) 
    return m

def main():
    m=cube(1);
    m.save_to_obj_file("C:/Users/nguye/Desktop/out.obj")

if __name__ == '__main__':
    main()
#!/usr/bin/python

import random
import pdb
import time

class Mesh(object): 
    
	vertex_count=0
	
    def __init__(self): 
        self.vertex_dict = dict() # Takes a vertex, returns the index of it. 
        self.face_list = [] # faces are lists. These lists contain vertex indices.
    
    def add_vertex(self, x,y,z):
        if (x,y,z) not in self.vertex_dict:
            self.vertex_dict[(x,y,z)]= vertex_count+1 
            vertex_count += 1
    
    def add_face(self, v_list):
        for v in v_list:
            self.add_vertex(v)
        self.face_list.append([self.vertex_dict[v] for v in v_list])
    
    def save_to_obj_file(self, output_file):
        with open(output_file,'w') as f:
            f.write("# Vertices\n")
            for i,(coordinate,index) in enumerate(sorted(self.vertex_dict.items(),key=lambda x:x[1])):
                assert i == index
                f.write("v "+str(coordinate[0])+" "+str(coordinate[1])+" "+str(coordinate[2])+"\n") 
            f.write("\n\n")
            f.write("# Faces\n")
            for face in self.face_list:
                f.write("f "+[str(v_index)+" " for v_index in face]+"\n")

def main():
    mesh = Mesh()
    mesh.add_face([(0,0,0),(1,0,0),(0,1,0)])
    mesh.save_to_obj_file("./out.obj")

if __name__ == '__main__':
    main()
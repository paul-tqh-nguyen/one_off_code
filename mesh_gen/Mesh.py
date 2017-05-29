#!/usr/bin/python

import random
import pdb
import time
from math import cos, sin, pi, sqrt

X_COORD=0
Y_COORD=1
Z_COORD=2

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

def mean(l):
    return sum(l)/float(len(l))

def square(x):
    return x*x

def euclidean_distance(vector):
    return sqrt(sum([square(e) for e in vector]))

def tuple_subtraction(A,B):
    return tuple(map(lambda e:e[0]-e[1],zip(A,B)))

def tuple_addition(A,B):
    return tuple(map(lambda e:e[0]+e[1],zip(A,B)))

def dot_product(A,B):
    return tuple(map(lambda e:e[0]*e[1],zip(A,B)))

def normalize_vector(v):
    return tuple(map(lambda x:x/euclidean_distance(v),v))

def cross_product(*args): 
    if len(args)==2:
        ax = args[0][X_COORD]
        ay = args[0][Y_COORD]
        az = args[0][Z_COORD]
        bx = args[1][X_COORD]
        by = args[1][Y_COORD]
        bz = args[1][Z_COORD]
    elif len(args)==6:
        ax = args[0]
        ay = args[1]
        az = args[2]
        bx = args[3]
        by = args[4]
        bz = args[5]
    else: 
        assert False
    return (ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx)

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

def get_equation_of_bisecting_circle(p1,p2,p3,radius):
    # https://math.stackexchange.com/questions/73237/parametric-equation-of-a-circle-in-3d-space 
    p1_to_p2 = tuple_subtraction(p2,p1)
    p3_to_p2 = tuple_subtraction(p2,p3)
    cp = cross_product(p1_to_p2,p3_to_p2)
    if cp != (0,0,0): # i.e. if the 3 points do not form a line
        mean_vector = normalize_vector(tuple_addition(p1,p3)) # Note that this is not the bisecting coplanar vector (that's the variable "a" as we define it below)
        v = mean_vector # axis of rotation for the circle
        a = normalize_vector(cp) # unit vector perpendicular to axis
    else:
        v = p1_to_p2 # since the 3 points form a line, that line is the axis of rotation, so we can just choose either of the two vectors we calculated above
        arbitrary_vector = tuple_addition(v,(1,0,0)) # an arbitrary vector
        a = cross_product(v,arbitrary_vector) # we just want the variable a to be perpendicular to v, since the cross product of v and any arbitrary vector is perpendicular to both v and that arbitrary vector, this value works for a.
    b = cross_product(a,v) # unit vector perpendicular to axis
    c = p2 # point on axis
    
    ans_func = lambda theta: (
                    c[X_COORD]+radius*cos(theta)*a[X_COORD]+radius*sin(theta)*b[X_COORD],
                    c[Y_COORD]+radius*cos(theta)*a[Y_COORD]+radius*sin(theta)*b[Y_COORD],
                    c[Z_COORD]+radius*cos(theta)*a[Z_COORD]+radius*sin(theta)*b[Z_COORD],
                    )
    return ans_func

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
    # segment_precision refers to the number of rectangles used per segment (this is precision along the other dimension)
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

def horn(precision=360):
    m = Mesh()
    points = [
                (0,0,0),
                (0,0,1),
                (0,1,2),
                (0,2,3),
                ]
    radii = [
            None, # no radius bc it's the tip
            0.5,
            1,
            None, # no radius bc it's the last point that we use the determine the orientation of the second to last circle
            ]
    def get_circle_equation_for_point_index(point_index):
        assert point_index > 0 # can't be the first point
        assert point_index < len(points)-1 # can't be the last point
        return get_equation_of_bisecting_circle(points[point_index-1],points[point_index],points[point_index+1],radii[point_index])
    circle_equations = [None]+[get_circle_equation_for_point_index(i) for i in range(1,len(points)-1)]+[None]
    # we want our circles to connect and look like cylinders
    # thus, we should make the faces that connect in such a way that makes them look like "straight" cylinders rather than one that's been twisted and thus kind of look like twisted towels
    # we do this by making the triangles that go between the two circles as "flat" and "straight" as possible. 
    # we aim to do this heuristically (which means we're too lazy to do it properly and bc we want it to be fast, but this also means that we still may get some twisted cylinders in there)
    # we'll start with the point at the tip. This will be our first focus point. It'll connect to the point nearest to it in the next circle. Those will define the first triangle to go between thw two.
    # we'll incrementally in order along the same direction on the circles create the rest of the triangles
    # the point that the tip was cclosest to is the next focus point. we repeat until we're done. 
    # the flaw is that we can still get some "twisted" cylinders if the circles are placed a certain way (think about a circle flat on the XY plane with a z-coordinate of zero adn a circle flat on the XY plane that's very far away from the first circle with a z-coordinate of 1. If we draw triangles between the two circles this way, there may be some triangles that cross each other, which gives the twisted look. 
    cirlce_point_lists_by_point_index = []
    current_focus_point = points[0]
    for equation in circle_equations:
        if equation is None:
            cirlce_point_lists_by_point_index.append(None)
            continue
        points_in_circle = [equation(2*pi*precision_index/float(precision)) for precision_index in range(precision)]
        nearest_point = None
        nearest_point_index = None
        closest_dist=float("inf")
        for i,p in enumerate(points_in_circle): # we'll use the manhattan distance for speed
            m_dist = sum(tuple_addition(tuple(map(abs,p)),tuple(map(abs,current_focus_point))))
            if m_dist<closest_dist:
                closest_dist = m_dist
                nearest_point_index = i
                nearest_point = p
        current_focus_point = nearest_point
        new_points = points_in_circle[nearest_point_index:]+points_in_circle[:nearest_point_index]
        cirlce_point_lists_by_point_index.append(new_points)
    cirlce_point_lists_by_point_index.append(None)
    # Tip of horn to first circle
    tip_point = points[0]
    first_circle_points = cirlce_point_lists_by_point_index[1]
    for point_index, point in enumerate(first_circle_points):
        next_point = first_circle_points[(point_index+1)%len(first_circle_points)]
        m.add_face([
                    tip_point,
                    point,
                    next_point,
                    ])
    # Middle tube pieces
    for tube_index in range(1,len(points)-2):
        first_points = cirlce_point_lists_by_point_index[tube_index]
        second_points = cirlce_point_lists_by_point_index[tube_index+1]
        for precision_index in range(precision):
            m.add_face([
                        second_points[(precision_index+1)%precision],
                        first_points[(precision_index+1)%precision],
                        first_points[precision_index],
                        ])
            m.add_face([
                        second_points[precision_index],
                        second_points[(precision_index+1)%precision],
                        first_points[precision_index],
                        ])
            
    '''
    circ_1 = get_equation_of_bisecting_circle((0,0,0),(0,0,1),(0,1,1),1)
    circ_2 = get_equation_of_bisecting_circle((0,0,1),(0,1,1),(0,2,1),1)
    
    faces = list()
    for precision_index in range(precision):
        start_angle = 2*pi*precision_index/float(precision)
        end_angle = 2*pi*(precision_index+1)/float(precision)
        triangle_face_1 = [
                            circ_func_2(end_angle),
                            circ_func_1(end_angle),
                            circ_func_1(start_angle),
                        ]
        triangle_face_2 = [
                            circ_func_2(start_angle),
                            circ_func_2(end_angle),
                            circ_func_1(start_angle),
                        ]
        faces.append(triangle_face_1)
        faces.append(triangle_face_2)
    
    tube_faces = get_faces_for_tube(circ_1, circ_2, precision)
    for e in tube_faces:
        m.add_face(e)
    '''
    return m

def main():
    # Test code
    #m=cube()
    #m=cone()
    #m=torus()
    m=horn()
    m.save_to_obj_file("C:/Users/nguye/Desktop/out.obj")
    

if __name__ == '__main__':
    main()
#!/usr/bin/python

import sys
import os
import random
import copy
import scipy
import numpy
import Image
import matplotlib.pyplot

X_COORD = 0
Y_COORD = 1

def get_mutated_child(init_schematic,ni,nj,nk):
    schematic = copy.deepcopy(init_schematic)
    
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    range_x = range(ni)
    range_y = range(nj)
    z_start = random.randint(0,nk)
    random.shuffle(range_x)
    random.shuffle(range_y)
    identifier = 0
    for z in xrange(0,nk):
        for y in range_y:
            for x in range_x:
                if (z < z_start):
                    identifier = max(identifier, schematic[x][y][z])
                elif (schematic[x][y][z] == -1):
                    continue
                else:
                    identifier += 1
                    schematic[x][y][z] = identifier
    
    for z in xrange(z_start,nk):
        random.shuffle(directions)
        for y in range_y:
            for x in range_x:
                if (schematic[x][y][z] == -1):
                    continue
                (slot_is_double, (x_dir,y_dir)) = is_double(x,y,z,schematic,ni,nj,nk)
                if (not slot_is_double):
                    directions.append(directions.pop(0))
                    for e in directions:
                        if (x+e[X_COORD] >= 0 and x+e[X_COORD] < ni and y+e[Y_COORD] >= 0 and y+e[Y_COORD] < nj):
                            if (schematic[x+e[X_COORD]][y+e[Y_COORD]][z] == -1):
                                continue;
                            if is_single(x+e[X_COORD],y+e[Y_COORD],z,schematic,ni,nj,nk):
                                if (z != 0):
                                    if (schematic[x+e[X_COORD]][y+e[Y_COORD]][z-1] == schematic[x][y][z-1] and schematic[x][y][z-1] != -1):
                                        continue
                                schematic[x+e[X_COORD]][y+e[Y_COORD]][z] = schematic[x][y][z]
                                break
    return schematic
    
def connect_blocks(init_schematic,ni,nj,nk):
    schematic = copy.deepcopy(init_schematic)
    
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    range_x = range(ni)
    range_y = range(nj)
    range_z = range(nk)
    random.shuffle(range_x)
    random.shuffle(range_y)
    for z in range_z:
        random.shuffle(directions)
        for y in range_y:
            for x in range_x:
                if (schematic[x][y][z] == -1):
                    continue
                (slot_is_double, (x_dir,y_dir)) = is_double(x,y,z,schematic,ni,nj,nk)
                if (not slot_is_double):
                    directions.append(directions.pop(0))
                    for e in directions:
                        if (x+e[X_COORD] >= 0 and x+e[X_COORD] < ni and y+e[Y_COORD] >= 0 and y+e[Y_COORD] < nj):
                            if (schematic[x+e[X_COORD]][y+e[Y_COORD]][z] == -1):
                                continue;
                            if is_single(x+e[X_COORD],y+e[Y_COORD],z,schematic,ni,nj,nk):
                                if (z != 0):
                                    if (schematic[x+e[X_COORD]][y+e[Y_COORD]][z-1] == schematic[x][y][z-1] and schematic[x][y][z-1] != -1):
                                        continue
                                schematic[x+e[X_COORD]][y+e[Y_COORD]][z] = schematic[x][y][z]
                                break
    return schematic

def get_fitness(matrix,ni,nj,nk):
    num_single_blocks = 0
    num_valid_blocks = 0
    
    x_len = len(matrix)
    y_len = len(matrix[0])
    z_len = len(matrix[0][0])
    
    for z in xrange(z_len):
        for y in xrange(y_len):
            for x in xrange(x_len):
                if (matrix[x][y][z] != -1):
                    num_valid_blocks += 1
                if (is_single(x,y,z,matrix,ni,nj,nk)):
                    num_single_blocks += 1
    return 1.0-float(num_single_blocks)/num_valid_blocks

def CreatePNGSchematic(m_):
    matrix = copy.deepcopy(m_)
    x_len = len(matrix)
    y_len = len(matrix[0])
    z_len = len(matrix[0][0])
    
    grid_size = 25
    
    w = x_len*grid_size
    h = y_len*grid_size
    
    color_dict = {-1:(0,0,0)}
    
    for z in xrange(z_len):
        data = numpy.zeros( (h,w,3), dtype=numpy.uint8)
        for y in xrange(y_len):
            for x in xrange(x_len):
                if (matrix[x][y][z] not in color_dict.keys()):
                    color_dict[matrix[x][y][z]]=(int(random.random()*256),int(random.random()*256),int(random.random()*256))
                color_r = color_dict[matrix[x][y][z]][0]
                color_g = color_dict[matrix[x][y][z]][1]
                color_b = color_dict[matrix[x][y][z]][2]
                
                for r in xrange(y*grid_size,(y+1)*grid_size):
                    for c in xrange(x*grid_size,(x+1)*grid_size):
                        if (r%grid_size == 0 or c%grid_size == 0):
                            data[r,c] = [255,255,255]
                        else:
                            data[r,c] = [color_r,color_g,color_b]
        img = Image.fromarray(data, 'RGB')
        img.save('schematic'+str(z)+'.png')
    
def paulprint(l, print_to_screen=True): #my method of printing a 3d matrix
    x_len = len(l)
    y_len = len(l[0])
    z_len = len(l[0][0])
    
    final_output = '['
    for z in xrange(z_len):
        final_output += ('\n [' if z != 0 else '[')
        for y in xrange(y_len):
            final_output += ('  [' if y != 0 else '[')
            for x in xrange(x_len):
                if (l[x][y][z] == -1):
                    final_output += '     , '
                else:
                    final_output += '%5d'%(l[x][y][z])+', '
            final_output += ('],\n' if (y < y_len-1) else ']]\n')
    final_output += ']\n'
    if (print_to_screen):
        print final_output
    return final_output

def is_single(x0, y0, z0, matrix, ni, nj, nk): 
    if (matrix[x0][y0][z0] == -1):
        return False
    return not is_double(x0, y0, z0, matrix, ni, nj, nk)[0]

def is_double(x0, y0, z0, matrix, ni, nj, nk):
    if (matrix[x0][y0][z0] == -1):
        return (False, (0,0))
    if (x0+1<ni):
        if (matrix[x0][y0][z0] == matrix[x0+1][y0][z0]):
            return (True, (1,0))
    if (x0-1>=0):
        if (matrix[x0][y0][z0] == matrix[x0-1][y0][z0]):
            return (True, (-1,0))
    if (y0+1<nj):
        if (matrix[x0][y0][z0] == matrix[x0][y0+1][z0]):
            return (True, (0,1))
    if (y0-1>=0):
        if (matrix[x0][y0][z0] == matrix[x0][y0-1][z0]):
            return (True, (0,-1))
    return (False, (0,0))
def usage():
    print
    print "usage: python voxelizer.py <sdf_file_location> <output_directory> <num_chromosomes> <num_generations>"
    print 
    sys.exit(1)
    
def main():
    if (len(sys.argv) < 5):
        usage()
    
    sdf_file_location = os.path.abspath(sys.argv[1])
    output_directory = os.path.abspath(sys.argv[2])+'/'
    num_chromosomes = int(sys.argv[3])
    num_generations = int(sys.argv[4])
        
    try:
        os.makedirs(output_directory)
    except:
        pass
    
    if (not os.path.isdir(output_directory)):
        print "Problem with output_directory ("+output_directory+")."
        sys.exit(1)
    
    sdf_text = open(sdf_file_location, 'r').read().split('\n')
    (ni,nj,nk) = map(int, sdf_text.pop(0).split())
    (origin_x,origin_y,origin_z) = map(float, sdf_text.pop(0).split())
    dx = int(sdf_text.pop(0).split()[0])
    
    initial_schematic = [[[-1 for k in xrange(nk)] for j in xrange(nj)] for i in xrange(ni)]
    
    i_index = 0
    j_index = 0
    k_index = 0
    out_file_text = ''
    identifier = 100 # start at 100 to not get confused with -1 values
    k=origin_z
    while k < nk+origin_z:
        j_index = 0
        j = origin_y
        while j < nj+origin_y:
            i_index = 0
            i=origin_x
            while i < ni+origin_x:
                current_signed_distance = float(sdf_text.pop(0))
                if (current_signed_distance <= 0):
                    initial_schematic[i_index][j_index][k_index] = identifier
                    out_file_text += str(i)+' '+str(j)+' '+str(k)+'\n'
                    identifier += 1
                i += dx
                i_index += 1
            j+=dx
            j_index += 1
        k+=dx
        k_index += 1
    
    voxel_file = open(output_directory+'voxel_positions.voxel','w')
    voxel_file.write(out_file_text)
    voxel_file.close()
    
    # Genetic Search
    chromosomes = []
    for which_chromosome in xrange(num_chromosomes):
            chromosomes.append(connect_blocks(initial_schematic,ni,nj,nk))
    for which_generation in xrange(num_generations):
        fitness_values = []
        for gene in chromosomes:
            fitness_values.append( (get_fitness(gene,ni,nj,nk), gene) )
        fitness_values = sorted(fitness_values, key=lambda x: 1.0-x[0])
        print "Best Fitness in Generation "+str(which_generation)+":", fitness_values[0][0] #, ":", map(lambda x: round(x[0],3), fitness_values)
        next_generation = []
        for i in xrange(len(fitness_values)):
            if (i%2==0): #keep half as elite
                next_generation.append(copy.deepcopy(fitness_values[i/2][1]))
            else: #keep half as mutations of elite
                next_generation.append( get_mutated_child(fitness_values[(i-1)/2][1],ni,nj,nk) )
        chromosomes = copy.deepcopy(next_generation)
    schematic = copy.deepcopy(chromosomes[0]) # our final product
    print "Final Fitness:", get_fitness(schematic,ni,nj,nk)
    # End GS
    
    # connect all remaining adjacent singles
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    range_x = range(ni)
    range_y = range(nj)
    range_z = range(nk)
    for z in range_z:
        for y in range_y:
            for x in range_x:
                if (is_single(x,y,z,schematic,ni,nj,nk)):
                    for e in directions:
                        if (x+e[X_COORD] >= 0 and x+e[X_COORD] < ni and y+e[Y_COORD] >= 0 and y+e[Y_COORD] < nj):
                            if is_single(x+e[X_COORD],y+e[Y_COORD],z,schematic,ni,nj,nk):
                                schematic[x+e[X_COORD]][y+e[Y_COORD]][z] = schematic[x][y][z]
                                break
    
    # Write files
    schematic_copy = copy.deepcopy(schematic)
    for z in range_z:
        for y in range_y:
            for x in range_x:
                if (is_single(x,y,z,schematic,ni,nj,nk)):
                    schematic_copy[x][y][z] *= -1
    schematic = copy.deepcopy(schematic_copy)
    
    singles_text = ''
    doubles_text = ''
    for z in range_z:
        for y in range_y:
            for x in range_x:
                if (schematic[x][y][z] == -1): # if block not there
                    continue
                if (schematic[x][y][z] < 0): # if block is a single
                    singles_text += str(x)+' '+str(y)+' '+str(z)+'\n'
                    
                (slot_is_double, e) = is_double(x,y,z,schematic,ni,nj,nk)
                if (slot_is_double): # if block is a double
                    doubles_text += str(x)+' '+str(y)+' '+str(z)+' '
                    assert (schematic[x][y][z] == schematic[x+e[X_COORD]][y+e[Y_COORD]][z])
                    doubles_text += str(x+e[X_COORD])+' '+str(y+e[Y_COORD])+' '+str(z)
                    doubles_text += '\n'
    
    #paulprint(schematic)
    
    singles_file = open(output_directory+'lego_positions.singles','w')
    singles_file.write(singles_text)
    singles_file.close()
    
    doubles_file = open(output_directory+'lego_positions.doubles','w')
    doubles_file.write(doubles_text)
    doubles_file.close()
    
    CreatePNGSchematic(schematic)
    
if __name__ == '__main__':
    main()



#!/usr/bin/python

import os
import sys
import tkFileDialog
from gui import *
import copy
import time
import random
import ntpath
import subprocess
import Image
import numpy
import ImageDraw
import ImageFont
import operator

B_COORD = 0
A_COORD = 1

DEFAULT_DELTA = 50

LINE_THICKNESS = 10

ALPHA_NUMERIC_CHARACTERS = [chr(i) for i in xrange(0,256) if chr(i).isalnum()]
SKIP_CODE = 8

color_dict = {0:(random.randint(0,255),random.randint(0,255),random.randint(0,255))}
displayed_solution_index = 0
displayed_piece_index = 0
solution_images_to_display = []
piece_images_to_display = []

xxx_start = time.time()

paul_hash_list = {0:0}

def flatten(list_of_lists):
    return reduce(operator.add, list_of_lists)

def paul_hash(element):
    for k,v in paul_hash_list.items():
        if v==element:
            return k
    k = max(paul_hash_list.keys())+1
    paul_hash_list[k] = element
    return k

def paul_unhash(index):
    return paul_hash_list[index]

def convert_solution_to_piece(sol):
    final_piece = []
    for sol_index, (piece_index,l) in enumerate(sol):
        for tile in l:
            final_piece.append( (tile[0],tile[1],piece_index) )
    return final_piece

def update_color_dict(index):
    global color_dict
    if index < max(color_dict.keys()):
        return
    for i in xrange(index):
        if i not in color_dict.keys():
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            while color in color_dict.values():
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            color_dict[i] = color

def clear():
    os.system('clear')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def generate_random_name(length=100):
    return ''.join([random.choice(ALPHA_NUMERIC_CHARACTERS) for i in xrange(length)])

def generate_unique_file_name(length=100,extension='.py'):
    random_name = os.path.abspath(generate_random_name(length-len(extension))+extension)
    while os.path.isfile(random_name):
        random_name = os.path.abspath(generate_random_name(length-len(extension))+extension)
    return random_name

def generate_unique_directory_name(length=100):
    random_name = os.path.abspath(generate_random_name(length))
    while os.path.isdir(random_name):
        random_name = os.path.abspath(generate_random_name(length))
    return random_name

def print_piece_main(piece0,ascii_text,indentation_length=0):
    if len(piece0[0]) != 3: # if not normalized
        piece = normalize_piece_main(piece0,ascii_text)
    else:
        piece = copy.deepcopy(piece0)
    piece_height = 1+max(map(lambda e:e[B_COORD],piece))
    piece_width = 1+max(map(lambda e:e[A_COORD],piece))
    string = ''
    for y in xrange(piece_height):
        string += ' '*indentation_length+'['
        for x in xrange(piece_width):
            c = ' '
            for yy,xx,cc in piece:
                if y==yy and x==xx:
                    c = cc
            string +=  c
        string += ']\n'
    print string[:-1]

def print_matrix(m):
    w = len(m[0])
    h = len(m)
    string = ''
    for y in xrange(h):
        string += '['
        for x in xrange(w):
            string += str(m[y][x])
        string += ']\n'
    print string[:-1]

def sort_piece(piece):
    max_x = max(map(lambda e:abs(e[A_COORD]), piece))
    max_y = max(map(lambda e:abs(e[B_COORD]), piece))
    big_num = max_x*max_y
    return sorted(piece, key=lambda e: e[B_COORD]*big_num+e[A_COORD])

def normalize_piece_main(piece,ascii_text):
    noralized_piece = []
    min_x = min(map(lambda e:e[A_COORD], piece))
    min_y = min(map(lambda e:e[B_COORD], piece))
    for i,tile in enumerate(piece):
        noralized_piece.append( (tile[B_COORD]-min_y, tile[A_COORD]-min_x, ascii_text[tile[B_COORD]][tile[A_COORD]]) )
    return noralized_piece

def reflect_piece(piece):
    # reflects piece across horz axis
    reflected_piece = []
    for i in xrange(len(piece)):
        reflected_piece.append( (-piece[i][B_COORD],piece[i][A_COORD],piece[i][2]) )
    min_x = min(map(lambda e:e[A_COORD], reflected_piece))
    min_y = min(map(lambda e:e[B_COORD], reflected_piece))
    for i in xrange(len(reflected_piece)):
        reflected_piece[i] = (reflected_piece[i][B_COORD]-min_y,reflected_piece[i][A_COORD]-min_x,reflected_piece[i][2])
    return sort_piece(reflected_piece)

def rotate_piece(piece,num_rotations):
    rotated_piece = list(piece)
    for i in xrange(len(rotated_piece)):
        for ii in xrange(num_rotations):
            rotated_piece[i] = (rotated_piece[i][A_COORD],-rotated_piece[i][B_COORD],rotated_piece[i][2])
    min_x = min(map(lambda e:e[A_COORD], rotated_piece))
    min_y = min(map(lambda e:e[B_COORD], rotated_piece))
    for i in xrange(len(rotated_piece)):
        rotated_piece[i] = (rotated_piece[i][B_COORD]-min_y,rotated_piece[i][A_COORD]-min_x,rotated_piece[i][2])
    rotated_piece = sort_piece(rotated_piece)
    return rotated_piece

def evaluate_solution(pieces, goal_height, goal_width, goal_dict, potential_solution_description, output_directory):
    '''
    Solution descriptions are represented as dictionaries.
    The key is the indexinto pieces. 
    Each value is a tuple containing:
        the y position in the final output
        the x position in the final output
        the number of rotations of the piece to do before placing it down on the board
    If we have a problem, we'll return the index of the piece causing the problem. Problems can be pieces on top of each other or pieces being off the board.
    This info should be used to determine which branch of solutions to not use.
    Returns False if everything fits in but the solution isn't correct (e.g. if the colors don't match)
    Returns True on success.
    '''
    num_tiles = 0
    for k,(y, x, num_rotations) in potential_solution_description.items(): # if we don't have the same number of tiles as the goal, it won't work
        if num_rotations != SKIP_CODE:
            num_tiles += len(pieces[k])
    if num_tiles != len(goal_dict.keys()):
        return False
    current_solution = {} # solutions are represented as dictionaries where the keys are the coordinates and the values are 3-tuples containing (y_pos, x_pos, char)
    for i, piece in enumerate(pieces):
        y, x, num_rotations = potential_solution_description[i]
        if num_rotations == SKIP_CODE: # num_rotations == SKIP_CODE, this is just a hack to say that we don't use this piece
            continue
        rotated_piece = list(piece)
        if num_rotations > 3:
            num_rotations -= 4
            rotated_piece = reflect_piece(rotated_piece)
        rotated_piece = rotate_piece(rotated_piece,num_rotations)
        for dy,dx,c in rotated_piece:
            k = (y+dy,x+dx)
            if k not in goal_dict.keys() or goal_dict[k] != (y+dy,x+dx,c) or k in current_solution.keys() or y+dy >= goal_height or x+dx >= goal_width:
                return i
            current_solution[k] = (y+dy,x+dx,c)
    return goal_dict==current_solution

def generate_image(pieces, final_pieces, width, height, output_file_name):
    global color_dict
    delta = DEFAULT_DELTA
    image_width = width*delta+LINE_THICKNESS
    image_height = height*delta+LINE_THICKNESS
    font = ImageFont.truetype("./font.ttf",20)
    img=Image.new("RGBA", (image_width,image_height),"white")
    draw = ImageDraw.Draw(img)
    for i in xrange(width+1):
        for ii in xrange(LINE_THICKNESS):
            draw.line([i*delta+ii, 0, i*delta+ii, image_height-1], fill='grey')
    for i in xrange(height+1):
        for ii in xrange(LINE_THICKNESS):
            draw.line([0, i*delta+ii, image_width-1, i*delta+ii], fill='grey')
    draw = ImageDraw.Draw(img)
    for i,piece in final_pieces:
        for y,x,c in piece:
            color = color_dict[i]
            for ii in xrange(LINE_THICKNESS, delta):
                draw.line([x*delta+LINE_THICKNESS, y*delta+ii, (x+1)*delta, y*delta+ii], fill=color)
            draw.text(((x+0.4)*delta, (y+0.4)*delta),c,'black',font=font)
    draw = ImageDraw.Draw(img)
    img.save(output_file_name)

def solver(A, B, solution=[]):
    if not A:
        yield list(solution)
    else:
        minimum = min(A, key=lambda e: len(A[e]))
        for r_prime in list(A[minimum]):
            solution.append(r_prime)
            columns = allow(A, B, r_prime)
            for s in solver(A, B, solution):
                yield s
            disallow(A, B, r_prime, columns)
            solution.pop()

def allow(A, B, r_prime):
    columns = []
    for i_B in B[r_prime]:
        for i_A in A[i_B]:
            for i_B_B in B[i_A]:
                if i_B_B != i_B:
                    A[i_B_B].remove(i_A)
        columns.append(A.pop(i_B))
    return columns

def disallow(A, B, r, columns):
    for i_B in reversed(B[r]):
        A[i_B] = columns.pop()
        for i_A in A[i_B]:
            for i_B_B in B[i_A]:
                if i_B_B != i_B:
                    A[i_B_B].append(i_A)

def generate_piece_images(pieces, output_directory):
    global color_dict
    piece_image_names = []
    for piece_index, piece in enumerate(pieces):
        name=os.path.join(output_directory,path_leaf(generate_unique_file_name(extension='.png')))
        width = 1+max(map(lambda e:e[A_COORD], piece))
        height = 1+max(map(lambda e:e[B_COORD], piece))
        delta = DEFAULT_DELTA
        image_width = width*delta+LINE_THICKNESS
        image_height = height*delta+LINE_THICKNESS
        font = ImageFont.truetype("./font.ttf",20)
        img=Image.new("RGBA", (image_width,image_height),"white")
        draw = ImageDraw.Draw(img)
        for i in xrange(width+1):
            for ii in xrange(LINE_THICKNESS):
                draw.line([i*delta+ii, 0, i*delta+ii, image_height-1], fill='grey')
        for i in xrange(height+1):
            for ii in xrange(LINE_THICKNESS):
                draw.line([0, i*delta+ii, image_width-1, i*delta+ii], fill='grey')
        for y,x,c in piece:
            color = color_dict[piece_index]
            for ii in xrange(LINE_THICKNESS, delta):
                draw.line([x*delta+LINE_THICKNESS, y*delta+ii, (x+1)*delta, y*delta+ii], fill=color)
            draw.text(((x+0.4)*delta, (y+0.4)*delta),c,'black',font=font)
        draw = ImageDraw.Draw(img)
        img.save(name)
        piece_image_names.append(name)
    return piece_image_names

def select_file(gui, entry_handle):
    puzzle_location = tkFileDialog.askopenfilename()
    if len(puzzle_location) > 0:
        gui.get_widget(entry_handle).delete(0, Tkinter.END)
        gui.get_widget(entry_handle).insert(0, puzzle_location )

def calculate_goal(gui, text_label_handle, puzzle_location_entry_box_handle, output_directory):
    update_label = lambda x: gui.update_text_label(text_label_handle, x) # label updating code
    try:
        puzzle_location = gui.get_text_from_entry_box(puzzle_location_entry_box_handle)
        puzzle_location = os.path.abspath(puzzle_location) # get absolute path of puzzle ascii file
        update_label("Searching for solution for "+puzzle_location)
        if not os.path.isfile(puzzle_location): # make sure file exists
            update_label("The selected file does not exist. Please select a valid file.")
            return
        ascii_text = [ f for f in open(puzzle_location,'rt').read().split('\n') ] # access elements of ascii_text via ascii_text[y][x]
        # helper functions
        def normalize_piece(piece):
            return normalize_piece_main(piece,ascii_text)
        def print_piece(piece0, indentation_length=0):
            print_piece_main(piece0,ascii_text,indentation_length=indentation_length)
        # pad each line
        longest_length_line = max(map(len,ascii_text))
        for i,e in enumerate(ascii_text):
            pad_len = longest_length_line-len(e)
            if pad_len > 0:
                ascii_text[i] = e+' '*pad_len
        # parse ascii_text for pieces
        pieces = []
        width = longest_length_line
        height = len(ascii_text)
        piece_parts = []
        for y in xrange(height): # find piece parts
            for x in xrange(width):
                if ascii_text[y][x] != ' ':
                    piece_parts.append( (y,x) )
        while len(piece_parts)>0: # while there are parts left
            piece = [piece_parts.pop()] # start a new piece
            new_connecting_piece_part_found = True
            while new_connecting_piece_part_found: # while we're still looking to connect all pieces to the current piece 
                new_connecting_piece_part_found = False
                for i,(y,x) in enumerate(piece_parts): # go through all remainin parts 
                    for yy,xx in piece:                # to see if any are connected to the current piece
                        if abs(x-xx)<=1 and abs(y-yy)<=1:
                            piece.append(piece_parts.pop(i))
                            new_connecting_piece_part_found = True
                            break
                    if new_connecting_piece_part_found:
                        break
            pieces.append(piece)
        pieces = sorted(pieces, key=lambda x:-len(x)) # So we put down the bigger pieces first
        # sort each piece by y then by x
        for i, piece in enumerate(pieces):
            pieces[i] = sort_piece(piece)
        # Normalize each piece so that it's relative to (0,0)
        for i in xrange(len(pieces)):
            pieces[i] = normalize_piece(pieces[i])
            pieces[i] = sort_piece(pieces[i])
        # Remove the largest piece because that one's the goal piece
        max_length = 0
        goal_index = 0
        for i, length in enumerate(map(len, pieces)):
            if max_length < length:
                max_length = length
                goal_index = i
        goal = pieces.pop(goal_index)
        goal_height = 1+max(map(lambda e:e[B_COORD],goal))
        goal_width = 1+max(map(lambda e:e[A_COORD],goal))
        # Exhaustively evaluate solutions
        goal_dict = {}
        for y,x,c in goal:
            k = (y,x) 
            goal_dict[k] = (y,x,c)
        # Brute Force
#        potential_solution_description = {}
#        for i in xrange(len(pieces)):
#            potential_solution_description[i] = (0,0,0)
#        def increment_solution_description(solution_description, index=None):
#            global SKIP_CODE
#            MAA_B = goal_height-1
#            MAA_A = goal_width-1
#            MAA_R = SKIP_CODE # SKIP_CODE means don't use this piece
#            if index == None:
#                index = len(pieces)-1 # the last element 
#            if index < 0: # if we kept incrementing until we couldn't any more (similar to overflow), we've exhausted all possible solutions
#                return None
#            if solution_description[index][2] < MAA_R:
#                solution_description[index] = (solution_description[index][0],solution_description[index][1],solution_description[index][2]+1)
#                return solution_description
#            
#            if solution_description[index][1] < MAA_A:
#                solution_description[index] = (solution_description[index][0],solution_description[index][1]+1,0)
#                return solution_description
#            
#            if solution_description[index][0] < MAA_B:
#                solution_description[index] = (solution_description[index][0]+1,0,0)
#                return solution_description
#            else:
#                solution_description[index] = (0,0,0)
#                return increment_solution_description(solution_description, index=index-1)
#        solutions = []
#        start = time.time()
#        while potential_solution_description is not None:
#            solution_value = evaluate_solution(pieces, goal_height, goal_width, goal_dict, potential_solution_description, output_directory)
#            if solution_value is True:
#                if potential_solution_description not in solutions:
#                    final_pieces = ['']*len(pieces)
#                    for piece_index,(y,x,num_rotations) in potential_solution_description.items():
#                        final_piece = copy.deepcopy(pieces[piece_index])
#                        if num_rotations == SKIP_CODE: # hack to mean skip this piece
#                            continue
#                        if num_rotations > 3:
#                            num_rotations -= 4
#                            final_piece = reflect_piece(final_piece)
#                        final_piece = rotate_piece(final_piece,num_rotations)
#                        for i,tile in enumerate(final_piece):
#                            final_piece[i] = (tile[B_COORD]+y,tile[A_COORD]+x,tile[2])
#                        final_pieces[piece_index]=final_piece
#                    if final_pieces not in solutions:
#                        solutions.append(final_pieces)
#            if type(solution_value) is int:
#                potential_solution_description = increment_solution_description(potential_solution_description, index=solution_value)
#            else:
#                potential_solution_description = increment_solution_description(potential_solution_description)
#        end = time.time()
        ###################################################################
        # Algorithm A
        start = time.time()
        # Get all positions for all pieces
        piece_positions = {}
        for i,piece in enumerate(pieces):
            possible_positions = []
            for y in xrange(goal_height):
                for x in xrange(goal_width):
                    for r in xrange(SKIP_CODE):
                        new_piece = list(piece)
                        num_rotations = r
                        if num_rotations > 3:
                            num_rotations -= 4
                            new_piece = reflect_piece(new_piece)
                        new_piece = rotate_piece(new_piece,num_rotations)
                        for ii,(yy,xx,cc) in enumerate(new_piece):
                            k = (y+yy,x+xx)
                            if yy+y>=goal_height or xx+x>=goal_width or k not in goal_dict.keys() or goal_dict[k] != (y+yy,x+xx,cc):
                                new_piece = None
                                break
                            new_piece[ii] = (yy+y,xx+x,cc)
                        if new_piece != None and new_piece not in possible_positions:
                            possible_positions.append( new_piece )
            piece_positions[i] = possible_positions
        A={}
        for goal_coordinate in goal_dict.values():
            A[goal_coordinate]=[]
            for piece_index,list_of_positions in piece_positions.items():
                for position_index, position in enumerate(list_of_positions):
                    if goal_coordinate in position:
                        A[goal_coordinate].append( paul_hash((piece_index,position_index)) )
        B = {}
        for piece_index,list_of_positions in piece_positions.items():
            for position_index, position in enumerate(list_of_positions):
                B[paul_hash((piece_index,position_index))] = piece_positions[piece_index][position_index]
        solutions0 = [ f for f in solver(A, B) ]
        solutions0 = filter(lambda a:len(a)==len(set(map(lambda e:e[0],map(paul_unhash,a)))), solutions0) # remove bad solutions where one tile is used many times
        end = time.time()
        solutions = []
        solutions_as_set = []
        for unformatted_solution in solutions0:
            formatted_solution = map(lambda e: (e[0],piece_positions[e[0]][e[1]]),map(paul_unhash,unformatted_solution))
            formatted_solution_as_piece = convert_solution_to_piece(formatted_solution)
            if set(formatted_solution_as_piece) not in solutions_as_set:
                solutions.append( formatted_solution )
                for ii in xrange(4):
                    formatted_solution_as_piece = rotate_piece(formatted_solution_as_piece,1)
                    solutions_as_set.append(set(formatted_solution_as_piece))
                formatted_solution_as_piece = reflect_piece(formatted_solution_as_piece)
                for ii in xrange(4):
                    formatted_solution_as_piece = rotate_piece(formatted_solution_as_piece,1)
                    solutions_as_set.append(set(formatted_solution_as_piece))
                    
        total_time = end-start
        update_label("Took "+str(total_time)+" seconds to find "+str(len(solutions))+" solution"+("s" if len(solutions)>1 else "")+" found for puzzle located at "+puzzle_location)
        # Save the solutions
        os.system('rm '+output_directory+'/*png 2> /dev/null')
        output_files = []
        for solution in solutions:
            output_file_name = os.path.join(output_directory, path_leaf(generate_unique_file_name(extension='.png')))
            generate_image(pieces, solution, goal_width, goal_height, output_file_name)
            output_files.append(output_file_name)
        generated_images_location_list = generate_piece_images(pieces, output_directory)
        
        # Save solution for records
        log_folder = puzzle_location+'_solutions'+('_no_reflections' if SKIP_CODE == 4 else '')
        os.system('mkdir '+log_folder+' 2> /dev/null')
        os.system('rm -rf '+log_folder+'/* 2> /dev/null')
        for i,e in enumerate([os.path.abspath(f) for f in generated_images_location_list]):
            os.system('cp '+e+' '+os.path.join(log_folder,'piece_'+str(i)+'.png') )
        for i,e in enumerate([os.path.abspath(f) for f in output_files]):
            os.system('cp '+e+' '+os.path.join(log_folder,'solution_'+str(i)+'.png') )
        with open(os.path.join(log_folder,"time.txt"),'wt') as f:
            f.write("Total Time: "+str(total_time))
        return output_files, generated_images_location_list
    except Exception as err:
        update_label("Error.")
        raise

def main():
    update_color_dict(20) #hack
    clear()
    output_directory = generate_unique_directory_name()
    def quit():
        os.system('rm -rf '+output_directory+' 2> /dev/null')
        print 
        print "Solver has finished running."
        print 
        exit()
    os.system('mkdir '+output_directory+' 2> /dev/null')
    gui = Window(width=1200, height=900)
    puzzle_location_entry_handle = gui.add_text_entry_box(default_text=os.path.abspath('./puzzles/trivial'), width=100, x_pos=10, y_pos=15)
    text_display_label_handle = gui.add_text_label(text="", x_pos=10, y_pos=50)
    puzzle_location_select_button_handle = gui.add_button(text="Select File", command=lambda : select_file(gui, puzzle_location_entry_handle), x_pos=820, y_pos=10)
    solution_description_label_handle = gui.add_text_label(text="Solution 0 of 0:", x_pos=10, y_pos=100)
    piece_description_label_handle = gui.add_text_label(text="Piece 0 of 0:", x_pos=10, y_pos=600)
    solution_image_display_label_handle = gui.add_image_label(image_location="", x_pos=10, y_pos=150)
    piece_image_display_label_handle = gui.add_image_label(image_location="", x_pos=10, y_pos=650)
    def display_next_solution():
        global solution_images_to_display
        global piece_images_to_display
        global displayed_solution_index
        if len(solution_images_to_display)==0:
            gui.update_text_label(solution_description_label_handle, 'Solution 0 of 0')
            name=os.path.join(output_directory,path_leaf(generate_unique_file_name(extension='.png')))
            Image.new("RGBA", (4,4),"white").save(name)
            gui.update_image_label(solution_image_display_label_handle,name)
            return
        displayed_solution_index = (displayed_solution_index+1) % len(solution_images_to_display)
        gui.update_image_label(solution_image_display_label_handle,solution_images_to_display[displayed_solution_index])
        gui.update_text_label(solution_description_label_handle, 'Solution '+str(1+displayed_solution_index)+' of '+str(len(solution_images_to_display)))
    def display_next_piece():
        global solution_images_to_display
        global piece_images_to_display
        global displayed_piece_index
        if len(piece_images_to_display)==0:
            gui.update_text_label(piece_description_label_handle, 'Piece 0 of 0')
            name=os.path.join(output_directory,path_leaf(generate_unique_file_name(extension='.png')))
            Image.new("RGBA", (4,4),"white").save(name)
            gui.update_image_label(piece_image_label_handle,name)
            return
        displayed_piece_index = (displayed_piece_index+1) % len(piece_images_to_display)
        gui.update_image_label(piece_image_display_label_handle,piece_images_to_display[displayed_piece_index])
        gui.update_text_label(piece_description_label_handle, 'Piece '+str(1+displayed_piece_index)+' of '+str(len(piece_images_to_display)))
    next_solution_button_handle = gui.add_button(text="Next Solution", command=display_next_solution, x_pos=950, y_pos=90)
    next_piece_button_handle = gui.add_button(text="Next Piece", command=display_next_piece, x_pos=950, y_pos=590)
    def get_answer(allow_reflections=True):
        global solution_images_to_display
        global piece_images_to_display
        global displayed_solution_index
        global displayed_piece_index
        global SKIP_CODE
        if allow_reflections:
            SKIP_CODE = 8
        else:
            SKIP_CODE = 4
        solution_images_to_display, piece_images_to_display = calculate_goal(gui, text_display_label_handle, puzzle_location_entry_handle, output_directory)
        if len(solution_images_to_display)>0:
            displayed_solution_index = 0
            displayed_piece_index = 0
            gui.update_image_label(solution_image_display_label_handle,solution_images_to_display[displayed_solution_index])
            gui.update_text_label(solution_description_label_handle, 'Solution 1 of '+str(len(solution_images_to_display)))
        else:
            display_next_solution()
            display_next_piece()
        gui.update_image_label(piece_image_display_label_handle,piece_images_to_display[displayed_piece_index])
        gui.update_text_label(piece_description_label_handle, 'Piece 1 of '+str(len(piece_images_to_display)))
    solve_puzzle_button_handle = gui.add_button(text="Solve Puzzle", command=get_answer, x_pos=915, y_pos=10)
    solve_puzzle_button_handle = gui.add_button(text="Solve Puzzle Without Reflections", command=lambda : get_answer(allow_reflections=False), x_pos=820, y_pos=45)
    exit_button_handle = gui.add_button(text="Exit", command=quit, x_pos=1022, y_pos=10)
    try:
        gui.start()
    finally:
        quit()

if __name__ == '__main__':
    main()


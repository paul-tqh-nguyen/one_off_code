#!/usr/bin/python

import sys
import os

def usage():
    print
    print "usage: python scale.py <obj_file_location> <scale_factor>"
    print 
    sys.exit(1)
    
def main():
    if (len(sys.argv) < 3):
        usage()
    
    obj_file_location = os.path.abspath(sys.argv[1])
    scale_factor = float(sys.argv[2])
    
    new_text = ''
    
    offset = 0 # need to add this to make sure everything is positive
    with open(obj_file_location) as f:
        for line in f:
            if (len(line.strip())>0 and line.strip()[0:2] == 'v '):
                offset = min(offset, min(map( float,line[1:-1].strip().split(' ') )))
    
    with open(obj_file_location) as f:
        for line in f:
            if (len(line.strip())>0 and line.strip()[0:2] == 'v '):
                #new_text += " ".join(map(lambda x: str(20+int(float(x)*scale_factor)) if x.replace('.','',1).isdigit() else x, line.strip().split(' '))) + '\n'
                new_text += "v "+" ".join(map(lambda x: str(int((float(x)-offset)*scale_factor)), (line[2:-1].split(' '))))+"\n"
            else:
                new_text += line
    
    # output directory will be same as that of input obj file\
    output_file = open(os.path.dirname(obj_file_location)+'/scaled.obj','w')
    output_file.write(new_text)
    output_file.close()
    #print os.path.dirname(obj_file_location)+'/scaled.obj'
    #print new_text

if __name__ == '__main__':
    main()


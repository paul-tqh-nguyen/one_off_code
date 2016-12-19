
import random
import os

def main():
    
    input_image_name = os.path.abspath('./z.png')
    script_name = os.path.abspath('./x.sh')
    
    settings_directory = os.path.abspath('./x/')
    output_directory = os.path.abspath('./y/')
    
    if (not os.path.isdir(settings_directory)): 
        os.makedirs(settings_directory)
    
    if (not os.path.isdir(output_directory)): 
        os.makedirs(output_directory)
    
    g = open('x.sh', 'w')
    for i in xrange(10000):
        f = open(os.path.join(settings_directory,str(i)+'.settings'), 'w')
        f.write('''mu 15.0
numETFIterations 3
delta_m 1.0
delta_n 1.0
sigma_m 1.0
sigma_c 1.0
p 0.99
binary_threshold 1.0
numFDOGIterations 2
kernel_dim 13
blur_kernel_sigma 2.0
sigma_e '''+str(64+i%400)+'''
r_e '''+str(4.0+int(i/400))+'''
sigma_g 4.0
r_g 4.0
numFBLIterations 2
''')
        f.close()
        g.write('./main '+os.path.join(settings_directory,str(i)+'.settings')+' '+input_image_name+' '+os.path.join(output_directory,'out'+str(i)+'.png '))
        
        if (i%12==0 and i != 0):
            g.write(';')
            g.write('\n')
            g.write('wait;\n')
        else:
            g.write('&')
        g.write('\n')
        
    g.close()
    
    print "Command to run: \n"
    print '\nclear; clear; make clean; make; chmod u+x '+script_name+'; ./'+script_name+'\n'

if __name__ == '__main__':
    main()

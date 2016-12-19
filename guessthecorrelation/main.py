#!/usr/bin/python

import gtk.gdk
import pdb
import Image
import numpy
import math
import scipy
import scipy.ndimage
import time
import ImageEnhance

def std_dev(l):
    mean = sum(l)/len(l)
    ans = 0
    for e in l:
        ans += (e-mean)*(e-mean)
    ans /= len(l)
    ans = math.sqrt(ans)
    return ans

def covariance(X,Y):
    mean_X = sum(X)/len(X)
    mean_Y = sum(Y)/len(Y)
    ans = 0
    for x,y in zip(X,Y):
        ans += (mean_X-x)*(mean_Y-y)
    ans /= len(X)
    return ans

def main():
    # Stolen from http://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux See the replies to the comment
    w = gtk.gdk.get_default_root_window()
    sz = w.get_size()
    pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
    pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,sz[0],sz[1])
    if (pb != None):
        start_time = time.time()
        I = Image.frombuffer("RGB", (pb.get_width(),pb.get_height()) , pb.get_pixels(), 'raw', 'RGB', pb.get_rowstride(), 1)
        I=ImageEnhance.Sharpness(I).enhance(2)
        I=numpy.array(I)
        downsample_factor = 4
        if I[180,405,1]>200:
            downsample_factor = 1
        I = I[283:601,323:647,0] # crop
        f1 = numpy.vectorize(lambda x: 255 if x>100 else x) # Remove bright gray lines
        f2 = numpy.vectorize(lambda x: 0 if 0<x and x<100 else 255)
        I = numpy.asarray( f2(f1(I)), dtype="uint8" ) # ensure correct type
        # Non-maximal Suppression
        I = I[::downsample_factor,::downsample_factor] # Downsample by some factor in both dimension
#        Image.fromarray(I).save("original.png","png")
        I[0,:]=255
        I[-1,:]=255
        I[:,0]=255
        I[:,-1]=255
        for y in xrange(1,I.shape[0]-1):
            for x in xrange(1,I.shape[1]-1):
                if I[y,x]==0 and 0 in [I[y-1,x+1], I[y-1,x], I[y-1,x-1], I[y,x+1], I[y,x-1], I[y+1,x+1], I[y+1,x], I[y+1,x-1]]:
                    I[y,x]=255
#        Image.fromarray(I).save("suppressed.png","png")
        y_values = []
        x_values = []
        for y in xrange(1,I.shape[0]-1):
            for x in xrange(1,I.shape[1]-1):
                if I[y,x]==0:
                    y_values.append(float(y)/I.shape[0])
                    x_values.append(float(x)/I.shape[1])
        correlation = covariance(x_values, y_values)/(std_dev(x_values)*std_dev(y_values))
        print '%0.2f' % correlation
        return correlation
    else:
        print "Unable to get the screenshot."

if __name__ == '__main__':
    main()


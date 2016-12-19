import os
import sys
import urllib
import re
import Image
import numpy
import random
import pdb

links = [f+'/sizes/o/' for f in '''
https://www.flickr.com/photos/blmiers2/6167391543
https://www.flickr.com/photos/127665714@N08/15531133598
https://www.flickr.com/photos/blmiers2/6134988568
https://www.flickr.com/photos/mar10os/1313480890
https://www.flickr.com/photos/metamerist/36353575
https://www.flickr.com/photos/73767885@N00/9085311239
https://www.flickr.com/photos/blavandmaster/9920975626
https://www.flickr.com/photos/marcinski/228211801
https://www.flickr.com/photos/neilsingapore/5357576472
https://www.flickr.com/photos/interiorphotos/3631847381
https://www.flickr.com/photos/concerttour/7155537134
https://www.flickr.com/photos/davidfloresm/6004566942
https://www.flickr.com/photos/concerttour/7155888292
https://www.flickr.com/photos/usagvicenza/8389396696
https://www.flickr.com/photos/solomonlieberman/3222509349
https://www.flickr.com/photos/borisv/4110030092
https://www.flickr.com/photos/89105118@N07/9984541613
https://www.flickr.com/photos/71574717@N00/8400839763
https://www.flickr.com/photos/piaser/3431504414
https://www.flickr.com/photos/usarmyafrica/3773164662
https://www.flickr.com/photos/usarmyafrica/3772221391
https://www.flickr.com/photos/mrtopf/4074083883
https://www.flickr.com/photos/itupictures/8365305296
https://www.flickr.com/photos/britishcouncilrus/15533056501
https://www.flickr.com/photos/rikomatic/9592967440
https://www.flickr.com/photos/tonibduguid/4919939430
https://www.flickr.com/photos/tonibduguid/4921027397
https://www.flickr.com/photos/tonibduguid/3900811524
https://www.flickr.com/photos/tonibduguid/3893030043/
https://www.flickr.com/photos/randomcliche/2536827529
https://www.flickr.com/photos/andrews-photography-94/7935288394
https://www.flickr.com/photos/newdimensionfilms/4296023051
https://www.flickr.com/photos/snowboardguides/8466012104
https://www.flickr.com/photos/heraldpost/5936470044
https://www.flickr.com/photos/kenleewrites/2624787339
https://www.flickr.com/photos/maxtm/2688129310
https://www.flickr.com/photos/usagapg/4479736738
https://www.flickr.com/photos/johnragai/14248699930
https://www.flickr.com/photos/matthewvenn/366986172
https://www.flickr.com/photos/eole/8340689237
https://www.flickr.com/photos/iamthebestartist/701545692
https://www.flickr.com/photos/walkadog/3574300516
https://www.flickr.com/photos/cindy47452/106319028
https://www.flickr.com/photos/otarako/15688796445
https://www.flickr.com/photos/defenceimages/8937595777
https://www.flickr.com/photos/skohlmann/11204950596
https://www.flickr.com/photos/pmiaki/6746249887
https://www.flickr.com/photos/stuckincustoms/2254516298
https://www.flickr.com/photos/dotbenjamin/2693526336
https://www.flickr.com/photos/sneeu/3076501061
https://www.flickr.com/photos/normalityrelief/3075723695
https://www.flickr.com/photos/56218409@N03/16018971290
https://www.flickr.com/photos/unitedsoybean/9632910180
https://www.flickr.com/photos/thomashawk/12611996505
https://www.flickr.com/photos/hansel5569/6992498071
https://www.flickr.com/photos/mazzali/5342336206
https://www.flickr.com/photos/salehalnemari/6890537251
https://www.flickr.com/photos/tevescosta/14264255351
https://www.flickr.com/photos/cost3l/15636122658
https://www.flickr.com/photos/sanmitrakale/9730067911
https://www.flickr.com/photos/araswami/2198753323
'''.split('\n') if len(f) > 5]

random.shuffle(links)
#links = links[:2]

valid_image_links = []

os.system('clear')
os.system('clear')

for i, link in enumerate(links):
    continue_going = False
    while (True):
        try:
            html_text = urllib.urlopen(link).read()
            print link
            continue_going = True
            break
        except Exception as err:
            print err
            pass
    if not continue_going:
        continue
    
    for e in html_text.split():
        x = re.search('(?<=src=").+(_o\.jpg)', e)
        if x != None:
            image_link = e.replace('src="','').replace('">','')
            valid_image_links.append(image_link)
            break
print

final_images = []

for i,image_link in enumerate(valid_image_links):
    cmd = 'wget --output-document '+str(i)+'.jpg '+image_link
    print cmd
    os.system(cmd)
    image_name = str(i)+'.jpg'
    I = numpy.array(numpy.asarray(Image.open(image_name)))
    I_h, I_w = I.shape[:2]
    if I_w < I_h*4.0/3.0:
        os.system('rm '+image_name)
        continue
    pad_length = int((I_w-I_h*4.0/3.0)/2.0)
    I = I[:,pad_length:I_w-pad_length,:]
    Image.fromarray(I).save(image_name)
    os.system('convert '+image_name+' -resize 1000x750 '+image_name)
    
    
    
    
    
    


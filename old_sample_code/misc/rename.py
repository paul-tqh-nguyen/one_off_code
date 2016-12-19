#!/usr/bin/python

# Script that renames everyfile in the cwd randomly. Mainly used for shuffling file order around. 
# I find that I use this script often, so I feel that others may also find it useful (hopefully).

import os
import hashlib

for i,e in enumerate(sorted([ f for f in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(os.getcwd(),f)) and f != os.path.split(__file__)[1]])):
    fileName, fileExtension = os.path.splitext(e)
    os.rename(e, str(hash(e))+fileExtension)


#!/usr/bin/python

import subprocess
import os
import time
import pdb

TIME_BETWEEN_KEYS = 0.01

def press_enter():
    os.system("xte 'key Return'")
    time.sleep(TIME_BETWEEN_KEYS)

def alt_tab():
    os.system("xte 'keydown Alt_L' 'key Tab' 'keyup Alt_L'")
    time.sleep(TIME_BETWEEN_KEYS)

def press_keys(keys, num_times=1):
    for i in xrange(num_times):
        cmd = 'xdotool key'
        for k in keys:
            cmd += " "+k
        os.system(cmd)
        print cmd

def main():
    alt_tab()
    press_keys('w',25)
    press_keys('op',50)
#    for i in xrange(5):
    while True:
        press_keys('qwopwop',1)
#        press_keys('qwwoopp',1)
    alt_tab()
#        press_keys(['space'],1)

if __name__ == '__main__':
    main()

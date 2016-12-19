#!/usr/bin/python

import subprocess
import os
import time
import pdb
import main as guesser

TIME_BETWEEN_KEYS = 0.2

def press_enter():
    os.system("xte 'key Return'")
    time.sleep(TIME_BETWEEN_KEYS)

def alt_tab():
    os.system("xte 'keydown Alt_L' 'key Tab' 'keyup Alt_L'")
    time.sleep(TIME_BETWEEN_KEYS)

def press_keys(keys):
    cmd = 'xte '
    for k in keys:
        cmd += "'key "+k+"' "
    os.system(cmd)
    time.sleep(TIME_BETWEEN_KEYS)

def main():
    keys_to_press = '%02d' % min(99,(int(round(abs(guesser.main()*100),0))))
    alt_tab()
    
    while True:
        keys_to_press = str(int(round(abs(guesser.main()*100),0)))
        press_keys(keys_to_press)
        press_enter()
#        time.sleep(0.10)
        press_enter()

if __name__ == '__main__':
    main()


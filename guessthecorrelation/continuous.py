#!/usr/bin/python

import subprocess
import os
import time

def main():
    all_child_subprocesses = []
    cmd = 'python main.py'
    all_child_subprocesses.append(subprocess.Popen(cmd, stdin=None, shell=True))
    time_last_job_added = time.time()
    while True:
        if time.time()-time_last_job_added>0.8: # Add a job every so many seconds
            all_child_subprocesses.append(subprocess.Popen(cmd, stdin=None, shell=True))
            time_last_job_added = time.time()
        if len(all_child_subprocesses)>8: # No more than so many jobs at a time
            all_child_subprocesses[0].wait()
            all_child_subprocesses.pop(0)

if __name__ == '__main__':
    main()


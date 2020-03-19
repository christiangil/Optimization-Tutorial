#This script submits jobs to ICS-ACI

import os
import os.path
import glob
import numpy as np

def submit_job(path, job_name):
    os.system('mv %s %s'%(path, job_name))
    os.system('qsub %s'%job_name)
    os.system('mv %s %s'%(job_name,path))

###############################

Njobs_counter = 0

script_dir = os.path.dirname(os.path.realpath(__file__))
jobs_dir = "jobs/"
os.chdir(jobs_dir)
job_names = glob.glob("*.pbs")
os.chdir(script_dir)
for job_name in job_names:
    print(job_name)
    path = jobs_dir + job_name
    submit_job(path, job_name)
    Njobs_counter += 1

print('found and submitted %d jobs'%(Njobs_counter))
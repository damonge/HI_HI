#####################################################################################
#This script is used to generate the folders, run_parameters files, and 
#job scripts and run them together when many L_PICOLA realizations are desired.

import numpy as np
import os
import time


################################## INPUT #######################################
number_of_files = 100    #number of realizations wanted

run_param_file  = 'run_parameters.dat'  #name of the fiducial run_parameters file
fiducial_seed = '5001'            #seed value in the fiducial seed

################################################################################


#do a loop over all realizations. The folder of first realization is 1, not 0
for i in xrange(1,number_of_files+1):

   #realization folder name and seed value
   folder = str(i);      seed = 5001 + i*10

   #create folders, in case they dont exist
   if not os.access(folder,    os.F_OK):  os.makedirs(folder)    

   #move the run_parameters file to the realization folder changing the seed number
   f_in  = open(run_param_file,'r')
   f_out = open(folder+'/'+run_param_file,'w')

   for line in f_in:
       f_out.write(line.replace(fiducial_seed,str(seed)))
   f_in.close(); f_out.close()

   #wait some time to run a group of 20 realizations because we can't submit 100 jobs at the same time (don't know why):
   if i%20==0 and i!=100:
      time.sleep(20)

   #copy the ref_script into folders
   os.system('cp ref_script.sh '+folder)

   os.system('cd '+folder+' && qsub ref_script.sh && cd ../')

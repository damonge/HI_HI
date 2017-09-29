#####################################################################################
#This script is used to generate the folders, run_parameters files, and 
#job scripts and run them together when many L_PICOLA realizations are desired.

import numpy as np
import os
import time


################################## INPUT #######################################
run_param_file  = 'analysis.py'  #name of the fiducial run_parameters file
fiducial_seed = 'I = 12 ; J = 7 '            #seed value in the fiducial seed
################################################################################

# boxes = np.array([11,12,2,3,4])
# boxes = np.array([4])
boxes = np.array([31,32,33])
patch = np.arange(0,5)

#do a loop over all realizations. The folder of first realization is 1, not 0

for i in boxes:
   for p in patch:

      print i, p
      #realization folder name and seed value
      folder = 'box'+str(i)+'_'+str(p);      seed = 'I = '+str(i)+' ; J = '+str(p)+' '

      #create folders, in case they dont exist
      if not os.access(folder,    os.F_OK):  os.makedirs(folder)    

      #move the run_parameters file to the realization folder changing the seed number
      f_in  = open(run_param_file,'r')
      f_out = open(folder+'/'+run_param_file,'w')

      for line in f_in:
          f_out.write(line.replace(fiducial_seed,str(seed)))
      f_in.close(); f_out.close()

      # #wait some time to run a group of 20 realizations because we can't submit 100 jobs at the same time (don't know why):
      # if i%20==0 and i!=100:
      #    time.sleep(20)

      #copy the ref_script into folders
      os.system('cp job_script.sh '+folder)

      os.system('cd '+folder+' && qsub job_script.sh && cd ../')

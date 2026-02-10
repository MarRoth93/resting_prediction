# predictiing task activation based on resting state data

## data usage
nsd dataset from aws 
subjects 1,2,5,7
nsdgeneral mask for visual areas only
the nsddataset has many a lot of data of subjects watching images. often the same image a couple of times. this data can be averaged to have a average activation pattern for each stimulus.  the scripts prepare_nsddata.py and download_nsddata.py ggive information on how the task data can be prepared and downloaded. 


## goal  
use resting state data from subjects to predict their fmri activation when they watch an image. 
bets case would be to train the model on subjects 1,2,5 and ten being able to predict the activation of subject 7 based on stimuli by just using the subjects resting state. so zero shot inference would be optimal, but using some examples from subject 7 for training would be fine as well. 

## big problem
i want to use nsdgeneral mask only to reduce the voxel number. but differet subjects have different amounts of voxel numbers.

# other details
look at this project (/home/marco/Marco/03_alignment/hyperalignment_resting_state), maybe some functions and ideas help.



# usage


# INPUT:
# generate_code_errors directory  calibrationSpecKey 
# direcotry is the folder that contains the images
# calibrationSpecKey (along with the smartphone model name) is part of the key into the calibration table

# 
# directory that contains the image data should also contain a comma delimited file (markers_metadata.csv) which describes
# the expected location of the markers

# OUTPUT
# 
# a file called "markerLocations.txt" is created in the folder that contains the locations of the markers for ground truthing
# debug images photos are also created that show the quality of the codes


# MISCELLANEOUS
# need to add a new entry to the distotionParams[] table for every additional (camera,calibration) combination




from __future__ import print_function

import numpy as np

import cv2

import sys
import os
import re



import pollTireStates
import tireProcessStateSettings


height=2988
width=5312
maskBorderSize=300
    
   
   
def generateMasksSmartPhoneMaskFiles(dirtotest):
    
    photoFiles = [f for f in os.listdir(dirtotest)  if re.match('I.*(JPG|jpg)',f)]
    imageMasked=np.empty( (height,width,3) ,dtype='uint8')
    imageMasked[:,:]=[255,255,255]
    
    imageMasked[:,0:maskBorderSize]=[0,0,0]
    imageMasked[:,width-maskBorderSize:width]=[0,0,0]
    imageMasked[0:maskBorderSize,:]=[0,0,0]
    imageMasked[height-maskBorderSize:height,:]=[0,0,0]
    
    
    # set up the partial mask here
#    #imageMasked[:,maskBorder:width-maskBorder]=[255,255,255]
#    imageMasked[maskBorder:height-maskBorder,:]=[255,255,255]
    
    
    for currentPhotoIndex in range(len(photoFiles)):
        fnname=dirtotest+"\\M"+photoFiles[currentPhotoIndex]
        cv2.imwrite(fnname,imageMasked)
        

    
    # returns the index of the border photo
    return(currentPhotoIndex)
        



def generateMasksSmartPhone(dirtotest):

 
    



    generateMasksSmartPhoneMaskFiles(dirtotest  )
    
    #print ("\n ****************", onlyTirePhotoIndexStartingFromLeft,onlyTirePhotoIndexStartingFromRight)
 
        






pollTireStates.pollTireTransactionStatus(generateMasksSmartPhone,  tireProcessStateSettings.stateSmartPhoneOnlyReadyForMaskProcessing,tireProcessStateSettings.stateSmartPhoneOnlyReadyFor3DProcessing,3)




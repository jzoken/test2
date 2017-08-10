

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




dirtotest="D:\\temp\delme1\\f\\"
dirtotest="D:\\temp\delme2\\inewclamp130mmHMaskNewEagle\\"
dirtotest="D:\\temp\\delme2\inewclamp130mmHMaskWornEagleInflatedReal\\"
dirtotest="D:\\temp\\delme2\\April26Clamp130mmByHandNewEagle\\"
dirtotest="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanAppA\\"
dirtotest="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanApp_2016-05-21_11-02-09\\"
dirtotest="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanApp_2016-05-21_14-43-28_tw\\"

height=2988
width=5312

xEndOfLeftTarget=530
xStartOfRightTarget=4680
borderSignalThreshold=400
   
#image = cv2.imread(filename)
# Convert BGR to HSV
#imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   

def CheckPhotoForTarget(startRow,endRow,imagehsv):
    foundTarget=False
    startingRow=-1
    for rowNumber in range(startRow,endRow,np.sign(endRow-startRow)):
       # maskRow  = (imagehsv[rowNumber,:,0]>30) & (imagehsv[rowNumber,:,0]<80) & (imagehsv[rowNumber,:,1]>25) & (imagehsv[rowNumber,:,1]<140) & (imagehsv[rowNumber,:,2]>160)
        #maskRow  = (imagehsv[rowNumber,:,0]>130) & (imagehsv[rowNumber,:,0]<175) & (imagehsv[rowNumber,:,1]>91) & (imagehsv[rowNumber,:,1]<178) & (imagehsv[rowNumber,:,2]>191) &  (imagehsv[rowNumber,:,2]<229)
        # Tom's tape below
        #maskRow  = (imagehsv[rowNumber,:,0]>30) & (imagehsv[rowNumber,:,0]<60) & (imagehsv[rowNumber,:,1]>96) & (imagehsv[rowNumber,:,1]<221) & (imagehsv[rowNumber,:,2]>160) &  (imagehsv[rowNumber,:,2]<255)
      # Scoth green masking tape (impressions)
        #maskRow  = (imagehsv[rowNumber,:,0]>55) & (imagehsv[rowNumber,:,0]<72) & (imagehsv[rowNumber,:,1]>108) & (imagehsv[rowNumber,:,1]<214) & (imagehsv[rowNumber,:,2]>50) &  (imagehsv[rowNumber,:,2]<190)
        maskRow  = (imagehsv[rowNumber,:,0]>63) & (imagehsv[rowNumber,:,0]<82) & (imagehsv[rowNumber,:,1]>90) & (imagehsv[rowNumber,:,1]<220) & (imagehsv[rowNumber,:,2]>165) &  (imagehsv[rowNumber,:,2]<226)


        borderSignal=np.count_nonzero(maskRow[xEndOfLeftTarget:xStartOfRightTarget])
        if (borderSignal>borderSignalThreshold):
            closestRowToTireWithSignal=rowNumber
            startingRow=closestRowToTireWithSignal
            foundTarget=True
            break;

    return(startingRow,foundTarget)
 
    
   
   
def generateMasksForSideAndFindBorder(leftSide,dirtotest):
    
    photoFiles = [f for f in os.listdir(dirtotest)  if re.match('I.*(JPG|jpg)',f)]
    imagehsv2bgrMasked=np.empty( (height,width,3) ,dtype='uint8')
    imagehsv2bgrMasked[:,:,:]=[255,255,255]
    numPhotos=len(photoFiles)
    if leftSide:
        startPhotoIndex=0
        endPhotoIndex=numPhotos
        startRowIndex=0
        endRowIndex=height-1
    else:
        startPhotoIndex=numPhotos-1
        endPhotoIndex=0
        startRowIndex=height-1
        endRowIndex=0
        
          
    
    #while not borderPhoto:
    for currentPhotoIndex in range(startPhotoIndex,endPhotoIndex,np.sign(endPhotoIndex-startPhotoIndex) ):
        
        # read the photo        
        fullname=dirtotest+"\\"+photoFiles[currentPhotoIndex]
        image = cv2.imread(fullname)        
        imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        startRowIndexForNextPhoto,foundTarget=CheckPhotoForTarget(startRowIndex,endRowIndex,imagehsv)
        
        if leftSide:
            sindex=startRowIndexForNextPhoto+1
            eindex=endRowIndex
        else:
            eindex=startRowIndexForNextPhoto+1
            sindex=endRowIndex
            
                                                                  
                                                                  
        if foundTarget:
            
            #imagehsv2bgrMasked[startRowIndexForNextPhoto+1:endRowIndex,xEndOfLeftTarget:xStartOfRightTarget]=[0,0,0]
            imagehsv2bgrMasked[sindex:eindex,xEndOfLeftTarget:xStartOfRightTarget]=[0,0,0]
            fnname=dirtotest+"\\M"+photoFiles[currentPhotoIndex]
            cv2.imwrite(fnname,imagehsv2bgrMasked)
            # set it back to white
           
            #imagehsv2bgrMasked[startRowIndexForNextPhoto+1:endRowIndex,xEndOfLeftTarget:xStartOfRightTarget]=[255,255,255]

            imagehsv2bgrMasked[sindex:eindex,xEndOfLeftTarget:xStartOfRightTarget]=[255,255,255]

            # write out mask
        else:
            break
        startRowIndex=startRowIndexForNextPhoto
    
    # returns the index of the border photo
    return(currentPhotoIndex)
        



def generateMasks(dirtotest):

 
    


    onlyTirePhotoIndexStartingFromLeft = generateMasksForSideAndFindBorder(True, dirtotest  )
    onlyTirePhotoIndexStartingFromRight = generateMasksForSideAndFindBorder(False, dirtotest  )
    
    print ("\n ****************", onlyTirePhotoIndexStartingFromLeft,onlyTirePhotoIndexStartingFromRight)
 
        






pollTireStates.pollTireTransactionStatus(generateMasks,  tireProcessStateSettings.stateReadyForMaskProcessing,tireProcessStateSettings.stateReadyFor3DProcessing,3)




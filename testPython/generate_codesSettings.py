# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""

import commonSettings


global markerInformation
markersMetadata = commonSettings.rootDir+"\\markersMetadata132mmTabloid.csv"
#markersMetadata = commonSettings.rootDir+"\\markersMetadata.csv"

global thresholdPixelDistanceFudgeFactor
thresholdPixelDistanceFudgeFactor = 300

global maxYInterceptGapForRow
maxYInterceptGapForRow = 20

global distortionAreaRemoveFudgeFactor
distortionAreaRemoveFudgeFactor=0.015

# related to Samsung Active
global pixelsPerMM
pixelsPerMM=33

global maxNumberOfPtsPerSide
maxNumberOfPtsPerSide=500

global maxSizeCodeBoxInPixels
maxSizeCodeBoxInPixels=300

global markerInformationPixelToCode
markerInformationPixelToCode= "markerInformation.txt"


# camera information for Samsung (currently hardcoded)
camera_metadeta_distortionParams_fx = 3.99411182e+03 
camera_metadeta_distortionParams_fy=3.99418122e+03 
camera_metadeta_distortionParams_cx=2.68713926e+03
camera_metadeta_distortionParams_cy=1.51055154e+03
camera_metadeta_distortionParams_k1=0.24503953  
camera_metadeta_distortionParams_k2=-0.80636859 
camera_metadeta_distortionParams_k3 =0.77637451

camera_metadeta_imageWidth=5312
camera_metadeta_imageHeight=2988
camera_metadeta_focalLength=4.3

camera_metadeta_sensorHeight=5.5

camera_target_config_distanceToObject=170










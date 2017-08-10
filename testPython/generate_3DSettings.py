# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""

import commonSettings



global photoScanGCPFile
#photoScanGCPFile= commonSettings.rootDir+"\\gcp.txt"
photoScanGCPFile= commonSettings.rootDir+"\\gcpTabloid.txt"

global photoScanCalibrationFile
photoScanCalibrationFile= commonSettings.rootDir+"\\psc_calibration.xml"

global photoScanProjectName
photoScanProjectName= "ps.psz"

global photoScanPlyName
photoScanPlyName = "tire.ply"


global photoScanLogName
photoScanLogName = "pslog.txt"

global photoScanDebugName
photoScanDebugName= "psdebug.txt"

global photoScanReprojectionErrorsName
photoScanReprojectionErrorsName= "photoScanReprojectionErrors.txt"

#
# choose from High, Medium. Low
global matchAccuracy
matchAccuracy="Medium"

# choose from Ultra, High, Medium. Low, Lowest
global modelAccuracy
modelAccuracy="Medium"

global faceCount
#faceCount=5000000
faceCount=3000000

# choose from High, Medium. Low
#global matchAccuracy
#matchAccuracy="High"
#
## choose from Ultra, High, Medium. Low, Lowest
#global modelAccuracy
#modelAccuracy="High"
#
#global faceCount
##faceCount=5000000
#faceCount=6000000
#









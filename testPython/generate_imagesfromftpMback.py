# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""


#

import os

import time
import sqlite3 as lite
import shutil
import glob

import generate_imagesfromftpSettings  
import databaseSettings
import tireProcessStateSettings

import commonSettings


            

def  ScanRootFolder ():
    
    print ("\n loop3 \n")
    
    # if database doesn't exist  then create it
    if  not os.path.isfile(databaseSettings.tireProcessStateDB):
        print ("aaa")
        con = lite.connect(databaseSettings.tireProcessStateDB)
        databaseSettings.createStateTransitionTable(con)
    else:
        print ("bbb")
        con = lite.connect(databaseSettings.tireProcessStateDB)
        
    
    print ("\n loop4 \n")
    while (True):
        print ("\n jjj ", generate_imagesfromftpSettings.incomingPath)
        topLevelDirs=[ name for name in os.listdir(generate_imagesfromftpSettings.incomingPath) if os.path.isdir(os.path.join(generate_imagesfromftpSettings.incomingPath, name)) ]
        print ("\n qqq", topLevelDirs)
        for topLevelDir in topLevelDirs:
            print ("\n top level dir ", topLevelDir)
            # make sure the FTP has stopped
            # wait till images have cooled off from the oven at least a bit
            #ftpMaybeInProgress = CheckIfFTPMaybeInProgress(generate_imagesfromftpSettings.incomingPath+"\\"+topLevelDir)
            # if possibility stil copying then skip folder and track in future pass
            
            print ("\n loop2 \n")
            tireScanFolder=generate_imagesfromftpSettings.incomingPath+"\\"+topLevelDir
            if not glob.glob(tireScanFolder+"\\"+"log.txt"):
                continue
 
        
            if not os.path.exists(commonSettings.toBeProcessedPath+"\\"+topLevelDir):
                shutil.move(generate_imagesfromftpSettings.incomingPath+"\\"+topLevelDir,commonSettings.toBeProcessedPath)
#                
#                if 'smart' in open(commonSettings.toBeProcessedPath+"\\"+topLevelDir+"\\"+"scan.txt").read():
#                    row = (tireProcessStateSettings.stateSmartPhoneOnlyReadyFor3DProcessing, topLevelDir)
#                else:
#                    row = (tireProcessStateSettings.stateReadyForCodeProcessing, topLevelDir)
                row = (tireProcessStateSettings.stateReadyForCodeProcessing, topLevelDir)
                databaseSettings.addEntry(con,row)

            # create the Debug directory
            debugDirectory=commonSettings.toBeProcessedPath+"\\"+topLevelDir+"\\"+commonSettings.debugName
            if not os.path.exists(debugDirectory):
                os.makedirs(debugDirectory)
            

        #sleep
                
        time.sleep(generate_imagesfromftpSettings.sleepTime)
             

ScanRootFolder()
                    
()        


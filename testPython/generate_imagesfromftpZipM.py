# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""


#

import os
import datetime
import re
import time
import sqlite3 as lite
import shutil
import glob

import generate_imagesfromftpSettings  
import databaseSettings
import tireProcessStateSettings

import commonSettings
import zipfile


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


            

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
        # for each zipfile in root directory, 
        #   1. extract to target dir (don't fail if already there) 
        #   2. add entry to database (don't fail if already ther)
        #   3. remove zipfile 
        zipScansToExtract=[ name for name in os.listdir(generate_imagesfromftpSettings.incomingPath) if name.endswith('zip')  ]
        print ("\n qqq", zipScansToExtract)
        for zipScanToExtract in zipScansToExtract:
            print ("\n top level dir ", zipScanToExtract)
            # if not a zipfile, then ignore
            with zipfile.ZipFile(generate_imagesfromftpSettings.incomingPath+"\\"+zipScanToExtract) as zf:
                zf.extractall(commonSettings.toBeProcessedPath)
            # 
                    
            #dtparsed=re.match("(\d\d\d\d)\-(\d\d)\-(\d\d)\_(\d\d)\-(\d\d)\-(\d\d)", zipScanToExtract)
            #print ("\n ** dt parsed it", dtparsed)
            row = (tireProcessStateSettings.stateReadyForCodeProcessing, zipScanToExtract[0:zipScanToExtract.rfind('zip')-1])
            a=0
            databaseSettings.addEntry(con,row)
            # delete original file
            os.remove(generate_imagesfromftpSettings.incomingPath+"\\"+zipScanToExtract)
            a=0
            
            #
        #sleep
                
        time.sleep(generate_imagesfromftpSettings.sleepTime)
             

ScanRootFolder()


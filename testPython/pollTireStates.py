# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:14:19 2015

@author: jzoken
"""

#

import os
import datetime
import re
import time
import sqlite3 as lite
import sys
import shutil

import commonSettings
import databaseSettings






             
def pollTireTransactionStatus(callback,waitingForState, nextState,sleepTime):
    while (True):
    
        
     
        con=lite.connect(databaseSettings.tireProcessStateDB)
      
        
        print ("\n connect is ",con)
       

        print ("\n ************ Locked **** \n")
        tireScanState=(waitingForState,)
        
        cnt=databaseSettings.checkCntTireScansInSpecifiedState(con,tireScanState)

        if (cnt>0):
            id,path = databaseSettings.getFirstTireScanInSpecifiedState(con,tireScanState) 
            
            # ignore path if non-exsistent - might happen if database out of sync with folders
            if not os.path.exists(commonSettings.toBeProcessedPath+"\\"+path):
                # set database to indicate non-existent poath
                databaseSettings.transitionToNextState(con,id,666)
                continue
            
            
            
            print("\n id path1", id, path,"\n")
            time.sleep(10)
          
            flog = open("c:\\temp\\testing.txt",'w')
            flog.write("\ntesting\n")
            flog.close()
            print ("\ntoBeProcessedPath+path", commonSettings.toBeProcessedPath+path )
            callback(commonSettings.toBeProcessedPath+"\\"+path)
            
            databaseSettings.transitionToNextState(con,id,nextState)
          
        else:
            print("\n no transactions waiting for proc\n")
        
        time.sleep(sleepTime)
      


    print("\n End of Pass")
    return
    

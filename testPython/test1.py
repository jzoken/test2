# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:15:59 2017

@author: jack
"""
import os
import datetime
import re
import time
import sqlite3 as lite
import sys
import shutil

import commonSettings
import databaseSettings



myvars = {}
with open("c:\\temp\\metadata.txt") as myfile:
    for line in myfile:
        name, var = line.partition(":")[::2]
        myvars[name.rstrip()] = var.strip()
        
myvars['SW']=int(myvars['SW'])
myvars['AR']=int(myvars['AR'])
myvars['WS']=int(myvars['WS'])

manufacturer=myvars['Manufacturer']
brand=myvars['Brand']
sw=myvars['SW']
ar=myvars['AR']
ws=myvars['WS']


print myvars
con=lite.connect(databaseSettings.tireProcessStateDB)
fr=databaseSettings.getFullTireRecord(con,manufacturer,brand,sw,ar,ws)
print ("\n fr ",fr)
        
a=0
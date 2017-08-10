# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""

import re


#def getEmailFromFile(filePath,metaDataFile):
#
#
#    
#    file = open(metaDataFile, "r")
#
#    for MAIL in file.readlines():
#        m = re.match( 'email:\s+(.+)', MAIL)
#        if m:
#            emailAddress=m.groups()
#            return(emailAddress[0])
#            
#
#def getTireModelFromFile(filePath,metaDataFile):
# 
#    file = open(metaDataFile, "r")
#
#    for TM in file.readlines():
#        m = re.match( 'tiremodel:\s+(.+)', TM)
#        if m:
#            tireModel=m.groups()
#            return(tireModel[0])
    
def getAttributeValueFromFile(filePath,attribute, metaDataFile):
 
    file = open(metaDataFile, "r")

    for AV in file.readlines():
        aval = re.match( attribute+':\s+(.+)', AV)
        if aval:
            attributeValue=aval.groups()
            return(attributeValue[0])
    return(attribute+" Not Provided")
    




"""
Created on Sat Oct 24 14:29:50 2015

@author: jzoken
"""


import sys
import os
import vtk
import imp
import generate_emailResultsSettings
import generate_reportsMSettings
import commonSettings

#import re

#import testpoll
import pollTireStates
import metaDataExtraction

import tireProcessStateSettings
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText
# import smtplib
from email.MIMEMultipart import MIMEMultipart

from os.path import basename
from email.mime.application import MIMEApplication


#        


def sendEmail(tiremodel,tiresize,reportFile):
    msg = MIMEMultipart()
    msg['From'] = generate_emailResultsSettings.fromUID
    msg['To'] = generate_emailResultsSettings.toUID
    msg['Subject'] = 'Your TireAudit Report'
    message = 'Attached is your TireAudit Report on ' + tiremodel + ' ' + tiresize
    msg.attach(MIMEText(message))



    
    file1=open(reportFile,"rb")
    part = MIMEApplication(
        file1.read(),
        Name=basename(reportFile)
    )
    part['Content-Disposition'] = 'attachment; filename="%s"' % basename(reportFile)
    msg.attach(part)

    
    mailserver = smtplib.SMTP('smtp.gmail.com',587)
    # identify ourselves to smtp gmail client
    mailserver.ehlo()
    # secure our email with tls encryption
    mailserver.starttls()
    # re-identify ourselves as an encrypted connection
    mailserver.ehlo()
    mailserver.login(generate_emailResultsSettings.fromUID,generate_emailResultsSettings.senderPassword)
    
    mailserver.sendmail(generate_emailResultsSettings.fromUID,generate_emailResultsSettings.toUID,msg.as_string())
    
    mailserver.quit()   
    return(0)            
    
         
    
    
def generate_emailResults(dirtotest):
    
   
#    reportname="TireAuditReport"
#  
#    reportfullpath=dirtotest+"\\"+reportname
#    #fp="D:\\temp\\TireAuditRoot\\TireScans\\2016-05-29_08-08-08\\tire.ply"
    
    metaDataFile= dirtotest+"\\"+ commonSettings.metaDataFile
   

    #emailAddress=metaDataExtraction.getAttributeValueFromFile(dirtotest,"email", metaDataFile)
    emailAddress="jzoken@pacbell.net"
    tiremodel="TireModel Not Provided"
    tiresize="TireSize Not Provided"
    print ("\n emailaddress is ", emailAddress)
    reportFile=dirtotest+"\\"+ generate_reportsMSettings.reportFile
    status=sendEmail(tiremodel,tiresize,reportFile)
    

    #DisplaySimpleReport1(orientedTireSwath,reportfullpath )



pollTireStates.pollTireTransactionStatus(generate_emailResults,  tireProcessStateSettings.stateReadyForEmailProcessing,tireProcessStateSettings.stateFinal,3)

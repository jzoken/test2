# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:36:15 2016

@author: jzoken
"""

import sys
import os
import vtk
import imp
import commonSettings
#import testpoll
import pollTireStates
import tireProcessStateSettings

#import tireauditroutines as ta

import random
import numpy as np
import math

import generate_reportsMSettings
import pollTireStates
import metaDataExtraction
import tireProcessStateSettings

import sqlite3 as lite


#from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage.interpolation import shift

import time

from PIL import Image,ImageFile

import databaseSettings

#

import requests
#import logging
import time
from bs4 import BeautifulSoup as bs
import json
from barely_json import parse
from fuzzywuzzy import fuzz
from operator import itemgetter

import numpy as np
import csv

import re







#databaseSettings = imp.load_source('databaseSettings', 'C:\\Projects\\tireProgramming\\testPython\\databaseSettings.py')
ta = imp.load_source('tireauditroutines', 'C:\\Projects\\tireProgramming\\lib\\tireauditroutines.py')


class TireSearch():
    def __init__(self):
        pass

    def max_val(slef,l, i):
        return max(enumerate(map(itemgetter(i), l)), key=itemgetter(1))

    def request_data(self,url,querystring):
        epoch_time = int(time.time())
        print epoch_time
        headers = {
            'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            'accept-encoding': "gzip, deflate, br",
            'accept-language': "en-US,en;q=0.8",
            'cache-control': "no-cache",
            'upgrade-insecure-requests': "1",
            'cookie':"s_lv="+str(epoch_time)+";",
            'user-agent': "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
        }

        response = requests.request("GET", url, headers=headers, params=querystring)
        return response

    def get_cnt(self):
        list_tiers=[]
        url = "https://www.tirerack.com/tires/TireSearchResults.jsp"

        querystring = {"zip-code": "",
                    "width": self.s1 +"/",
                    "ratio": self.s2,
                    "diameter":  self.s3,
                    "rearWidth": self.s1+"/",
                    "rearRatio": self.s2,
                    "rearDiameter": self.s3}

        response=self.request_data(url,querystring)

        soup = bs(response.text.encode('utf-8'), 'html.parser')
        cnt=soup.findAll('script',{'type':'text/javascript'})
        for jst in cnt:
            if 'tireList[i] =' in jst.text:
                up_cnt = jst.text.split('tireList[i] =')[1:]
                for x,element in enumerate(up_cnt):
                    try:
                        item=element.split("i++;")[0]
                        js_item=parse(item)
                    except:
                        continue
                    try:
                       tier_url_t='https://www.tirerack.com'+js_item['url'].replace("'","").encode('utf-8')
                    except:
                        tier_url_t=''
                    try:
                        partNumber=js_item['partNumber'].replace("'","").encode('utf-8')
                    except:
                        partNumber=''
                    try:
                        tireMake=js_item['tireMake'].replace("'","").encode('utf-8')
                    except:
                        tireMake=''
                    try:
                        tireModel=js_item['tireModel'].replace("'","").encode('utf-8')
                    except:
                        tireModel=''
                    try:
                        displaySize=js_item['displaySize'].replace("'","").encode('utf-8')
                    except:
                        displaySize=''
                    try:
                        tier_url=tier_url_t+'&partnum='+partNumber+'&vehicleSearch=false&fromCompare1=yes'
                    except:
                        tier_url=''

                    if self.brand==tireMake:
                        compare_result=fuzz.token_set_ratio(self.model,tireModel)
                    else:
                        compare_result=0
                    tier_list_item=[tireMake,tireModel,partNumber,displaySize,tier_url,compare_result]

                    list_tiers.append(tier_list_item)

        return list_tiers


    def get_tier_data(self,target_item_link, target_item_displaySize):
        #print  target_item_link, target_item_displaySize
        tier_response = self.request_data(target_item_link, {})
        soup = bs(tier_response.text.encode('utf-8'), 'html.parser')

        tier_cnt=soup.find('div',{'id':'allSpecs'}).findAll('div',{'class':'specificationSpacing'})[0]
        tier_cnt_data_head=tier_cnt.find('thead').find('tr').findAll('td')
        tier_cnt_data_body = tier_cnt.find('tbody').find('tr').findAll('td')
        last_x=len(tier_cnt_data_body)-1
        #print len(tier_cnt_data_head)  , len(tier_cnt_data_body)
        item_pack=[]
        for x,i in enumerate(tier_cnt_data_body):

            if x==0:
                try:
                    item_value=i.find('span',{'class':'currentSelect '}).contents[0]
                except:

                    item_value=i.contents[0]
            elif x==last_x:
                try:
                    item_value= i.text
                except:
                    item_value=''
            else:
                try:

                    item_value=i.find('a').contents[0]

                except:
                    item_value = i.text

            try:
                item_value=item_value.strip()
            except:
                item_value='ERROR'

            try:
                item_key= tier_cnt_data_head[x].text.strip()
            except:
                item_key="ERROR"

            item_pack_element=[item_key,item_value]
            item_pack.append(item_pack_element)

        return item_pack



    def job(self,search_data):
        self.brand=search_data[0]
        self.model = search_data[1]
        self.s1=str(search_data[2])
        self.s2 = str(search_data[3])
        self.s3 = str(search_data[4])

        tier_list=self.get_cnt()

        print 'results length',len(tier_list)
        if len(tier_list)==0:
            print 'error in receiving site content for this inputs'
            return ['999999']

        tier_match_best_result=self.max_val(tier_list, -1)

        print  'tier_match_best_result', tier_match_best_result

        if tier_match_best_result[1]<60:
            print "error hasn't found proper match"
            return ['888888']

        target_tier_item_id=int(tier_match_best_result[0])
        #print 'target_tier_item_id',target_tier_item_id
        target_item=tier_list[target_tier_item_id]
        target_item_link=target_item[4]
        target_item_displaySize=target_item[3]

        #test sample
        # target_item_link='https://www.tirerack.com/tires/tires.jsp?tireMake=Goodyear&tireModel=Eagle+LS&partnum=26SR6LS&vehicleSearch=false&fromCompare1=yes'
        # target_item_displaySize='P225/60R16'

        tier_data=self.get_tier_data(target_item_link, target_item_displaySize)
        return tier_data


#from TireTrack import TireSearch






def AddLogo(reportNameFullPath):
    print ("\n********* Adding logo")
   
    logoToUse=commonSettings.rootDir+"\\"+"logo250.png"
    print ("\n logotouse is ",logoToUse)
    reportImg=Image.open(reportNameFullPath)
    logoImg=Image.open(logoToUse)
    reportImg.paste(logoImg,(20,20))
    reportImg.save(reportNameFullPath)
    #reportImg.save("c:\\temp\\foo1.jpg")


def ProcessRaw3DClipToFrameNew(dirToTest):
    tname="tire.ply"
    tiredirToTest=dirToTest+"\\"+tname
   
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(tiredirToTest)
    reader.Update()
    pd = reader.GetOutput()
    
    # clip to the the inside of the code
    boundsRawMesh = pd.GetBounds()
#    yHigher=0.13
#    yLower=0
#    heightOfTargetBar=0.010
#    depthOfTargetBar=0.010
#    # ideally you should go low enough until the mnesh becomes crappy
#    maxTreadDepth=0.050
    
   # grooveSwath=ta.Threshold(orientedSwath,"DistanceIn32nds",1.0,999)
    
    pdLargeComponent=ta.KeepOnlyLargestConnectedComponent1(pd)
    
    filename=dirToTest+ "\\"+"OnlyLargestConnectComp.vtp"
    ta.LogVTK(pdLargeComponent, filename)
    
    
    #(xstartPos,xendPos)=FindXStartEndForClipping(pd)
    boundsInsideTargetBar=[ boundsRawMesh[0],boundsRawMesh[1],generate_reportsMSettings.yLower+generate_reportsMSettings.heightOfTargetBar+generate_reportsMSettings.slop,generate_reportsMSettings.yHigher-generate_reportsMSettings.heightOfTargetBar-generate_reportsMSettings.slop,boundsRawMesh[5]-generate_reportsMSettings.depthOfTargetBar-generate_reportsMSettings.maxTreadDepth ,boundsRawMesh[5]]
    clippedRawTireInsideTargetBar=ta.Clip(pdLargeComponent,boundsInsideTargetBar)
    
    clippedRawTireInsideTargetBar=ta.KeepOnlyLargestConnectedComponent1(clippedRawTireInsideTargetBar.GetOutput())
    
    filename=dirToTest+ "\\"+"clippedRawTireInsideTargetBar1.vtp"
    ta.LogVTK(clippedRawTireInsideTargetBar, filename)
    
    
    return(clippedRawTireInsideTargetBar)


def GetMetaData(dirToTest):
    
    tireIDPathOnly=dirToTest[dirToTest.rfind('\\')+1:len(dirToTest)]
    
    selectString1="SELECT  \
    Scan.ScanID,  \
    Scan.TireID,  \
    Scan.Path,  \
    Tire.DOTCode,   \
    Tire.Comments,  \
    Tire.NewWornFlag,  \
    Tire.MountedOnRim,  \
    Tire.MountedOnVehicle,   \
    TireType.BrandName,  \
    TireType.SectionWidth,  \
    Manufacturer.ManufacturerName,  \
    Tire.TireTypeID,  \
    TireType.PassengerCommercialFlag, \
    TireType.AspectRatio,\
    TireType.Radius \
    FROM Manufacturer , Scan INNER JOIN TireType ON Manufacturer.ManufacturerID = TireType.ManufacturerID   \
    INNER JOIN Tire ON TireType.TireTypeID = Tire.TireTypeID AND Tire.TireID = Scan.TireID \
    WHERE Scan.Path=" 
    selectString2= "\"" + tireIDPathOnly +"\""
    selectString3="ORDER BY \
    Tire.TireID ASC, \
    TireType.TireTypeID ASC, \
    Tire.NewWornFlag ASC "
    selectString=selectString1+selectString2+ " " +selectString3
    
    
    con = lite.connect(databaseSettings.tireProcessStateDB)
    
    cur=con.cursor()

    cur=con.cursor()
    cur.execute(selectString)
 

    item = cur.fetchone()
    
    if (item is None):
        return(0)
    
    return(item)

    

def ProcessRaw3DClipToFrame(dirToTest):
    tname="tire.ply"
    tiredirToTest=dirToTest+"\\"+tname
   
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(tiredirToTest)
    reader.Update()
    pd = reader.GetOutput()
    
    # clip to the the inside of the code
    boundsRawMesh = pd.GetBounds()
#    yHigher=0.13
#    yLower=0
#    heightOfTargetBar=0.010
#    depthOfTargetBar=0.010
#    # ideally you should go low enough until the mnesh becomes crappy
#    maxTreadDepth=0.050
    
    pdLargeComponent=ta.KeepOnlyLargestConnectedComponent1(pd)
    
    filename=dirToTest+ "\\"+"OnlyLargestConnectComp.vtp"
    ta.LogVTK(pdLargeComponent, filename)
    
    
    #(xstartPos,xendPos)=FindXStartEndForClipping(pd)
    boundsInsideTargetBar=[ boundsRawMesh[0],boundsRawMesh[1],generate_reportsMSettings.yLower+generate_reportsMSettings.heightOfTargetBar+generate_reportsMSettings.slop,generate_reportsMSettings.yHigher-generate_reportsMSettings.heightOfTargetBar-generate_reportsMSettings.slop,boundsRawMesh[5]-generate_reportsMSettings.depthOfTargetBar-generate_reportsMSettings.maxTreadDepth ,boundsRawMesh[5]]

    #boundsInsideTargetBar=[ boundsRawMesh[0],boundsRawMesh[1],generate_reportsMSettings.yLower+generate_reportsMSettings.heightOfTargetBar,generate_reportsMSettings.yHigher-generate_reportsMSettings.heightOfTargetBar,boundsRawMesh[5]-generate_reportsMSettings.depthOfTargetBar-generate_reportsMSettings.maxTreadDepth ,boundsRawMesh[5]]
    clippedRawTireInsideTargetBar=ta.Clip(pdLargeComponent,boundsInsideTargetBar)
    filename=dirToTest+ "\\"+"clippedRawTireInsideTargetBar.vtp"
    ta.LogVTK(clippedRawTireInsideTargetBar.GetOutput(), filename)
    
    return(clippedRawTireInsideTargetBar.GetOutput())




def NormalizeDistanceArrayBasedOnKnownScaleTreadWidth(pd,treadWidthInInches):
    
    print ("\n *** tread width ",treadWidthInInches )
    
    treadWidthInMeters=float(treadWidthInInches)/39.3
    # compute distance between high and low x
    for i in range(0,pd.GetPointData().GetNumberOfArrays()):
        if pd.GetPointData().GetArrayName(i)=="Distance":
            distIndex=i
            break
    
    distPts = pd.GetPointData().GetArray(distIndex)   
    numPts =  pd.GetNumberOfPoints()
    
    pdBounds=pd.GetBounds()
    treadWidthInMeshUnits=pdBounds[1]-pdBounds[0]
    #    adjustedDist = vtk.vtkFloatArray()
#    adjustedDist.SetNumberOfComponents(1)
#    adjustedDist.SetName("DistanceInMMs")
    

    for i in range (0,numPts):
        #print distCells    
        a0 = distPts.GetTuple(i)
      
#        distIn32nds = int(abs( a0[0]*1000.0 *25.0 / 32.0 - 0.5))
        distInMM = float(a0[0]*treadWidthInMeters/treadWidthInMeshUnits)
        
        distPts.SetTuple1(i,distInMM)
     
        

   
    

    


def Add32ndsToSwath(pd):
    
    for i in range(0,pd.GetPointData().GetNumberOfArrays()):
        if pd.GetPointData().GetArrayName(i)=="Distance":
            distIndex=i
            break
    
    distPts = pd.GetPointData().GetArray(distIndex)   
    numPts =  pd.GetNumberOfPoints()
#    vectors to 3 scalar arrays
#    

#    
     
    dataDist = vtk.vtkFloatArray()
    dataDist.SetNumberOfComponents(1)
    dataDist.SetName("DistanceIn32nds")
    
    

    for i in range (0,numPts):
        #print distCells    
        a0 = distPts.GetTuple(i)
      
#        distIn32nds = int(abs( a0[0]*1000.0 *25.0 / 32.0 - 0.5))
        distIn32nds = float(abs( a0[0]*1000.0 *32.0 / 25.0 ))
     
        

        dataDist.InsertValue(i,distIn32nds)
         
    pd.GetPointData().AddArray(dataDist)
    
    return(pd)


def ReturnMiddleProfile(pd):
    
    boundsSwath=pd.GetBounds()
    centerY= (boundsSwath[3]+boundsSwath[2])/2
     
    slicePos=centerY  
    planeOrientation=vtk.vtkPlane()
    planeOrientation.SetOrigin(0,slicePos,0)
    planeOrientation.SetNormal(0,1,0)
        
    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(planeOrientation)
    cutterA.SetInputData(pd)
    cutterA.Update()
    
    cutterAPD=cutterA.GetOutput()
    
    boundscutterAPD=cutterAPD.GetBounds()
    
    midProfileLine=vtk.vtkLineSource()
    midProfileLine.SetPoint1(boundscutterAPD[0]-0.01,boundscutterAPD[2],boundscutterAPD[5]+0.02)
    midProfileLine.SetPoint2(boundscutterAPD[1]+0.01,boundscutterAPD[2],boundscutterAPD[5]+0.02)
    midProfileLine.Update()
    midProfileLinePD=midProfileLine.GetOutput()
    
    ta.LogVTK(midProfileLine.GetOutput(),"c:\\temp\\line123.vtp")
    
    
    
    
    
    trabitinZ=vtk.vtkTransform()
    trabitinZ.Translate(0,0,0.0002)
    cutterAPD=ta.TransformPD(cutterAPD,trabitinZ )
    
    
    
    ta.LogVTK(cutterA.GetOutput(),"c:\\temp\\slice123.vtp")
    
    return(cutterA.GetOutput(),midProfileLinePD)
    
    

def ReturnMidTireProfileInformation(midTireSlicePD):
    

    
    fullslice_vtk_array= midTireSlicePD.GetPoints().GetData()
    fullslice_numpy_array = vtk_to_numpy(fullslice_vtk_array)
    
    numberOfPointArrays = midTireSlicePD.GetPointData().GetNumberOfArrays()
    #print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = midTireSlicePD.GetPointData().GetArrayName(i)
        if (aname=="DistanceIn32nds"):
            break

    
    distance_vtk_array=midTireSlicePD.GetPointData().GetArray(i)
    distance_numpy_array = vtk_to_numpy(distance_vtk_array)
    
    xyzDist=np.transpose(np.vstack((np.transpose(fullslice_numpy_array),distance_numpy_array)))
    xyzDistSortedIndex=np.argsort(xyzDist[:,0] )
    xyzDistSorted=xyzDist[xyzDistSortedIndex]
    
    
    # use < 1/32" for on surface
    xyzDistOnSurface=np.where( xyzDistSorted [:,3] < 1.0)[0]
    

       
    profile_numpy_array_argsort=xyzDistOnSurface
    profile_numpy_array_argsort_Shift=shift(profile_numpy_array_argsort,-1)
    
    gapSize=xyzDistSorted[profile_numpy_array_argsort_Shift ][:,0]- xyzDistSorted[profile_numpy_array_argsort ][:,0]
    
    gapsProfile = np.vstack((profile_numpy_array_argsort,profile_numpy_array_argsort_Shift,gapSize))
    
    
    #gapSize=gapsProfile[:,0][profile_numpy_array_argsort_Shift] - profile_numpy_array[:,0][profile_numpy_array_argsort]

    bigGaps=np.where(gapsProfile[:,:][2]>generate_reportsMSettings.grooveThresholdXToConsider)
    intervalsStart=gapsProfile[:,bigGaps][0][0].astype(np.integer)
    intervalsEnd=gapsProfile[:,bigGaps][1][0].astype(np.integer)
    

    intervals=np.vstack( (intervalsStart,intervalsEnd) )
    
#  
#    for item in np.nditer(intervals):
#        rng=np.arange( item
#        r1=xyzDist[item[0]:item[1],:]
#        a=0
    deepGrooveList=np.empty((0,4))
    for i1 in range( intervals.shape[1] ):
        r1=np.arange(intervals[0,i1],intervals[1,i1])
        deepestPartOfGroove=np.argmax(xyzDistSorted[r1,3])
        deepestPartOfGrooveXYZDist=xyzDistSorted[r1[deepestPartOfGroove]]
        deepGrooveList=np.vstack( (deepGrooveList,deepestPartOfGrooveXYZDist) )
        
     
  
    
    # return the slice and the polydata as result
    return (  deepGrooveList  )


def ReturnProfileActor(pd):
    
    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputData(pd)
    
    profileActor=vtk.vtkActor()
#    profileActor.GetProperty().SetColor(1, 0, 1)
#    profileActor.GetProperty().SetEdgeColor(1, 0, 1)
    profileActor.GetProperty().SetColor(0.8, 0.8, 0)
    profileActor.GetProperty().SetEdgeColor(0.8, 0.8, 0)

    profileActor.GetProperty().SetLineWidth(5)
    profileActor.GetProperty().EdgeVisibilityOn()
    profileActor.GetProperty().SetOpacity(1)
    profileActor.SetMapper(profileMapper)
    
    return(profileActor)



    
 
def GenerateTextAnnotationForGrooves(ypos,deepGroovePointList):
    # item in groove list is of the form y and (centerX,distance)
    textActorList=[]

    for i in range(len(deepGroovePointList)):
        #centerX,distance=(grooveList[0][i], grooveList[1][i] )
        
        textSource=vtk.vtkVectorText()
        distAsString = str( round(deepGroovePointList[i][3],1 ) ) 
        
        #distAsString = str(int(np.floor(deepGroovePointList[i][3]*1000)))
        textSource.SetText(distAsString)
        
        aLabelTransform = vtk.vtkTransform()
        aLabelTransform.Identity()
        #aLabelTransform.Translate(-0.2, 0, 1.25)
        #aLabelTransform.Translate(deepGroovePointList[i][0],0, 0)
        aLabelTransform.Translate(deepGroovePointList[i][0],deepGroovePointList[i][1]-0.003, 0)
        
        #aLabelTransform.Scale(0.005, .005, .005)
        aLabelTransform.Scale(0.003, .003, .003)
        
                # Move the label to a new position.
        labelTransform = vtk.vtkTransformPolyDataFilter()
        labelTransform.SetTransform(aLabelTransform)
        labelTransform.SetInputConnection(textSource.GetOutputPort())
        
        # Create a mapper and actor to display the text.
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(labelTransform.GetOutputPort())
        
        labelActor = vtk.vtkActor()
        labelActor.SetMapper(labelMapper)
        
        

        labelActor.GetProperty().SetColor ( 1,0,0)
  
        textActorList.append(labelActor)
    
    return(textActorList)
        
 

def GenerateTransformedSlice(midSlicePD,tireSwathPD) :
    
    rotatedsliceTransform = vtk.vtkTransform()
    midSlicePDBounds=midSlicePD.GetBounds()
    tireSwathPDBounds=tireSwathPD.GetBounds()
    

    # translate the slice to 0 y
    
    
    
    
    #rotatedsliceTransform.Identity()
    # translate so that the midpoint of the slice is at
    rotatedsliceTransform.Translate(0,-midSlicePDBounds[2], 0)
    rotatedsliceTransform.RotateX(270)
    rotatedsliceTransform.PostMultiply()
    # move it to top of the swath plus a little
    rotatedsliceTransform.Translate(0,tireSwathPDBounds[3]+0.001, 0)
    
    rotatedMidSlicePD=ta.TransformPD(midSlicePD,rotatedsliceTransform).GetOutput()
    filename="c:\\temp\\rotslice.vtp"
    ta.LogVTK(rotatedMidSlicePD,filename)
    # put back to 0.1


#    
#
#   # Create a mapper and actor to display the text.
#    rotatedsliceMapper = vtk.vtkPolyDataMapper()
#    rotatedsliceMapper.SetInputConnection(rotatedsliceTransformPDFilter.GetOutputPort())
#    
#    rotatedsliceActor = vtk.vtkActor()
#    rotatedsliceActor.SetMapper(rotatedsliceMapper)
  


    return(rotatedMidSlicePD)
   

def ExportAsPNG(reportNamePNG,renWin):
        
    
    #reportNamePNG=reportName+".png"
    #reportNameJPG=reportName+".jpg"
    
    vtkW2Image =vtk.vtkWindowToImageFilter()
    vtkW2Image.SetInputBufferTypeToRGB()
    vtkW2Image.SetInput(renWin)
    vtkW2Image.Update()
    
#    vtkJPEGWriter=vtk.vtkJPEGWriter()
#    vtkJPEGWriter.SetFileName(reportName)
#    
#    vtkJPEGWriter.SetInputData(vtkW2Image.GetOutput())
#    vtkJPEGWriter.Write()
    
    vtkPNGWriter=vtk.vtkPNGWriter()
    vtkPNGWriter.SetFileName(reportNamePNG)
    
    vtkPNGWriter.SetInputData(vtkW2Image.GetOutput())
    vtkPNGWriter.Write()
  
    


def ExportAsPDF(reportName,renWin):
        
    vtkGL2PSExporter =vtk.vtkGL2PSExporter()
    vtkGL2PSExporter.SetRenderWindow(renWin)

    vtkGL2PSExporter.SetFileFormatToPDF()
    filenameForPdf=reportName
    vtkGL2PSExporter.DrawBackgroundOn()
    vtkGL2PSExporter.SetFilePrefix(filenameForPdf)
    vtkGL2PSExporter.SetFileFormatToPDF();
    vtkGL2PSExporter.SetCompress(0)
    vtkGL2PSExporter.Write()    
    

def GenerateText (caption, position, fontSize, color):
    
    textActor=vtk.vtkTextActor()
    textActor.SetInput (caption)
    textActor.SetPosition ( position[0],position[1] );
    #textActor.GetTextProperty().SetJustificationToCentered	()
    textActor.GetTextProperty().SetFontSize ( fontSize );
    textActor.GetTextProperty().SetColor ( color[0],color[1],color[2])
    return( textActor )


        

def GenerateLegend(numItems, title, lowerLeft, upperRight, barRatio,lut):
    
    textProp=vtk.vtkTextProperty()
    textProp.SetItalic (1)
    textProp.SetFontSize (10)
    textProp.SetColor (0,0,0)
    
    titleTextProp=vtk.vtkTextProperty()
    titleTextProp.SetItalic (0)
    titleTextProp.SetJustificationToCentered	()	

    titleTextProp.SetFontSize (10)
    titleTextProp.SetColor (0,0,0)
    
    # create the scalar_bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetOrientation(0)
    scalar_bar.SetLookupTable(lut)
#    scalar_bar.SetTitle("My Title")
    
    scalar_bar.SetLabelFormat("%.0f")
    scalar_bar.SetLabelTextProperty(textProp)
    scalar_bar.SetTitleTextProperty(titleTextProp)
    
 
    
    #scalar_bar.SetDrawFrame(1)
    #scalar_bar.SetWidth(.30)
    #SetVerticalTitleSeparation	(	int 		)	
    
#    
    scalar_bar.SetPosition(lowerLeft[0],lowerLeft[1])
    scalar_bar.SetPosition2(upperRight[0],upperRight[1])
    #scalar_bar.SetUnconstrainedFontSizeOn()
    scalar_bar.SetBarRatio(barRatio)
    scalar_bar.SetNumberOfLabels(numItems)
    
    scalar_bar.SetTitle("Groove Depth (32nd inch)")
    
    scalar_bar.SetTitleRatio(0.5 )
    
    return(scalar_bar)


def GenLUT(low,high):
    # Create a custom lut. The lut is used both at the mapper and at the
    # scalar_bar
    lut = vtk.vtkLookupTable()
    lut.Build()
    lut.SetTableRange(low,high)
    return(lut)


def RenderActors(actorList,ren):
    
    for ac in actorList:
        ren.AddActor(ac)
        
  
def PrepareBlocksAndGroovesForRendering(blockSwathDecPD,grooveSwathDecPD, colorLookupTable, lutRange):
        
    blockSwathDecMapper = vtk.vtkPolyDataMapper()
    blockSwathDecMapper.SetInputData(blockSwathDecPD)
 
     
    blockSwathDecActor = vtk.vtkActor()
    blockSwathDecActor.SetMapper(blockSwathDecMapper)
    
    
    grooveSwathDecMapper = vtk.vtkPolyDataMapper()
    grooveSwathDecMapper.SetInputData(grooveSwathDecPD)
    grooveSwathDecMapper.SetScalarModeToUsePointData() 
    ##grooveSwathMapper.ScalarVisibilityOn()
    ##grooveSwathMapper.SetColorModeToMapScalars()
    grooveSwathDecMapper.SelectColorArray('DistanceIn32nds')
    grooveSwathDecMapper.SetScalarRange(lutRange[0],lutRange[1])
    grooveSwathDecMapper.SetLookupTable(colorLookupTable)
    
    grooveSwathDecActor = vtk.vtkActor()
    grooveSwathDecActor.SetMapper(grooveSwathDecMapper)
    
    return(blockSwathDecActor,grooveSwathDecActor)
        

def TranslateTireSwathToOriginXY(tireSwath):
    
    bounds=[0,0,0,0,0,0]
    bounds=tireSwath.GetBounds()
    
    xTranslate = -(bounds[1]-bounds[0])/2
    yTranslate =-(bounds[3]-bounds[2])/2
    
        
    orginTransform=vtk.vtkTransform()
    orginTransform.Translate( xTranslate,yTranslate,0)
    
    orientedTireSwath=ta.TransformPD(tireSwath,orginTransform).GetOutput()
      
    return(orientedTireSwath)
    

def ModifySwathForDisplay(orientedSwath):
     
    # decimate as neccessary
     
    
    transTireSwath = vtk.vtkTransform()
   
    transTireSwath.RotateX(-20)
    orientedSwath=ta.TransformPD(orientedSwath, transTireSwath).GetOutput()
        
    
    filename="c:\\temp\\tdinches.vtp"
    ta.LogVTK(orientedSwath, filename)
    

    #orientedSwath=ta.ThresholdTread(orientedSwath,-0.0005,1)
    orientedSwath.GetPointData().SetActiveScalars("RGB")
    
    # Separate grooves from tread blocks
    # kinds of confusing because we use 32nd and mm
    
    #### change to 
    grooveSwath=ta.Threshold(orientedSwath,"DistanceIn32nds",1.0,999)
    #grooveSwath=ta.Threshold(orientedSwath,"Distance",0.0006,1)
    grooveSwath.GetPointData().SetActiveScalars("DistanceIn32nds")
    
    filename="c:\\temp\\gs.vtp"
    ta.LogVTK(grooveSwath, filename)
    
    blockSwath=ta.Threshold(orientedSwath,"DistanceIn32nds",0.0,1.0)
    filename="c:\\temp\\bsthresh.vtp"
    ta.LogVTK(blockSwath, filename)
    

    
    #orientedSwathThreshDec=vtk.vtkQuadricDecimation()
    grooveSwathDec=vtk.vtkDecimatePro()
    grooveSwathDec.SetInputData(grooveSwath)
    grooveSwathDec.SetTargetReduction (0.25)
    grooveSwathDec.Update()
    filename="c:\\temp\\grooveSwathDec.vtp"
    ta.LogVTK(grooveSwathDec.GetOutput(), filename)
    

     
    blockSwathDec=vtk.vtkDecimatePro()
    blockSwathDec.SetInputData(blockSwath)
    blockSwathDec.SetTargetReduction (0.9)
    blockSwathDec.Update()
    filename="c:\\temp\\blockSwathdec.vtp"
    ta.LogVTK(blockSwathDec.GetOutput(), filename)
    
    return(blockSwathDec.GetOutput(),grooveSwathDec.GetOutput())


def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()



def GetMetaData1(dirPath):
        
    
    
    myvars = {}
    with open(dirPath+"\\"+"metadata.txt") as myfile:
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
    tw=databaseSettings.getFullTireRecord(con,manufacturer,brand,sw,ar,ws)
    print ("\n tw ",tw)
    
    return(manufacturer,brand,sw,ar,ws,tw)

def GetMetaData2(dirPath):
        
    
    
    myvars = {}
    with open(dirPath+"\\"+"metadata.txt") as myfile:
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
    
    ts=TireSearch()
    tw= ts.job([manufacturer,brand,sw,ar,ws])[9][1]
    tw=(tw[:len(tw)-1])
    
    print ("\n tw is in inches:", tw)
    
   
    
    return(manufacturer,brand,sw,ar,ws,tw)



def ProcessReportNew1(dirToTest):
    
    #manufacturer,brand,sw,ar,ws,tw=GetMetaData2(dirToTest)
    #GY Eagle tw 6.0"
    tw=6.7
    #tw=6.2
    
    #clippedRawTireInsideTargetBarPD=ProcessRaw3DClipToFrame(dirToTest) 
    
    

    
    #print ts.job(["Goodyear", "Eagle", 225, 60, 16])
    #print ts.job(["Goodyear", "Eaglee", 225, 60, 16])
    #print ts.job(["Continental", "ContiProContact", 215, 60, 16])
    #print ts.job(["Goodyear", "Assurance Fuel Max", 215, 55, 17])
        
#    filename=dirToTest+"\\"+"tireMetaData.txt"
#    tireMetaFile=open(filename, 'r')
#    tireMetaString=tireMetaFile.read()
#    fields=tireMetaString.split(",")
#    manufacturerName=fields[0]
#    brand=fields[1]
#    sectionWidth=fields[2]
#    aspectRatio=fields[3]
#    wheelSize=fields[4]
#    treadWidthInInches=fields[5]
    
    
    tname="tire.ply"
    tiredirToTest=dirToTest+"\\"+tname
    
    #manufacturer,brand,sw,ar,ws,tw=GetMetaData1(dirToTest)
    
   
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(tiredirToTest)
    reader.Update()
    clippedRawTireInsideTargetBarPD = reader.GetOutput()
    
    # just added 1/26/2016
    clippedRawTireInsideTargetBarPD=ta.KeepOnlyLargestConnectedComponent1(clippedRawTireInsideTargetBarPD)
    
    
    
    # aded
    
    pd=vtk.vtkPolyData()
    
    obb = vtk.vtkOBBTree()

    obb.SetDataSet(clippedRawTireInsideTargetBarPD)
    
    #obb.SetInputData(pd)
    #obb.AutomaticOn()
    obb.BuildLocator()
    obb.GenerateRepresentation (0, pd)
    filename=dirToTest+"\\"+"unorientedBB.vtp"
    ta.LogVTK(pd, filename)
 
    
    pcaTr=ta.PCAPD(pd)
    orientedBoxPD=ta.TransformPD(pd, pcaTr).GetOutput()
    orientedSwathPD=ta.TransformPD(clippedRawTireInsideTargetBarPD, pcaTr).GetOutput()
    
    # check to see the oriented swath isn't upside down
    # to do this clip a region in the top eigtht (z) middle(x)  or the bottom eigth (z) middle (x)
    # no points means it's upside down
    
    bdsOrientedSwathPD=orientedSwathPD.GetBounds()
    zDiff=(bdsOrientedSwathPD[5]-bdsOrientedSwathPD[4])
    xMid=(bdsOrientedSwathPD[0]+bdsOrientedSwathPD[1])/2   
    xCheck=0.02
    #checkClipOrientedSwathPD=ta.ClipPD(orientedSwathPD, (xMid-xCheck,xMid+xCheck,bdsOrientedSwathPD[2],bdsOrientedSwathPD[3],bdsOrientedSwathPD[4],bdsOrientedSwathPD[4]+zDiff/8) )
    checkClipOrientedSwathPD=ta.ClipPD(orientedSwathPD, (xMid-xCheck,xMid+xCheck,bdsOrientedSwathPD[2],bdsOrientedSwathPD[3],bdsOrientedSwathPD[5]-zDiff/8,bdsOrientedSwathPD[5] ) )

    if ( checkClipOrientedSwathPD.GetNumberOfPoints()==0 ):
        # flip swath around
        y180Transform=vtk.vtkTransform()
        y180Transform.RotateY(180)
        orientedSwathPD=ta.TransformPD(orientedSwathPD, y180Transform).GetOutput()
        
        
     #ta.LogVTK(orientedSwathPD, "c:\\temp\orientedSwathq.vtp")
    filename=dirToTest+"\\"+"orientedSwathBB.vtp"
    ta.LogVTK(orientedSwathPD, filename)
    filename=dirToTest+"\\"+"orientedBB.vtp"
    ta.LogVTK(orientedBoxPD, filename)
      
    clippedRawTireInsideTargetBarPD=orientedSwathPD
    
    filename=dirToTest+"\\"+"clippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(clippedRawTireInsideTargetBarPD, filename)
    

    numcuts=100
    bbox=[0,0,0,0,0,0]
    bbox=clippedRawTireInsideTargetBarPD.GetBounds()
#    
    minY=bbox[2]
    maxY=bbox[3]

    yInterval=(maxY-minY)/numcuts

    appendedSlicesForHull=vtk.vtkAppendPolyData()
#    

    bds=(-999,999,0,0,-999,999)
    yCurrent=minY
    
    smallHullDirectory=dirToTest+"\\"+"smallHullDirectory"
    if not os.path.exists(smallHullDirectory):
        os.makedirs(smallHullDirectory)
        
    chPD=ta.TopOfConvexHull2(clippedRawTireInsideTargetBarPD)
    #clipPD=ta.ClipPD1(clippedRawTireInsideTargetBarPD,(bds[0],bds[1],     yCurrent, yCurrent+yInterval ,    bds[4],bds[5]),True )  
    #chPD=ta.ConvexHullSciPy(clippedRawTireInsideTargetBarPD)
    filename=dirToTest+"\\"+ "chpd"  + ".vtp"
    ta.LogVTK(chPD,filename)
    
   
    trchPD= ta.TrimConvexHull2(chPD,-1,1,-0.9,0.9,0,1)
    topOfHull=trchPD



#    
#    for i in range(0,numcuts):
#     
#        clipPD=ta.ClipPD1(clippedRawTireInsideTargetBarPD,(bds[0],bds[1],     yCurrent, yCurrent+yInterval ,    bds[4],bds[5]),True )  
#        chPD=ta.ConvexHullSciPy(clipPD)
#        #trchPD=ta.TrimConvexHull(chPD)
#        #trchPD= ta.TrimConvexHull2(chPD,-1,1,-0.5,0.5,0,1)
#        trchPD= ta.TrimConvexHull2(chPD,-1,1,-0.9,0.9,0,1)
#        
#        filename=smallHullDirectory+ "\\" + "clitem_" +str(i)+"_.vtp"
#        ta.LogVTK(clipPD,filename)
# 
#        
#        filename=smallHullDirectory+ "\\" + "chitem_" +str(i)+"_.vtp"
#       
#        ta.LogVTK(chPD,filename)
#        
#        filename=smallHullDirectory+ "\\" + "trchitem_" +str(i)+"_.vtp"
#    
#        ta.LogVTK(trchPD,filename)
#        yCurrent=yCurrent+yInterval
#        
#        appendedSlicesForHull.AddInputData(trchPD)
#        appendedSlicesForHull.Update()
#        a=0
#    
#    filename="c:\\temp\\appendedSlicesForHull.vtp"
#    ta.LogVTK(appendedSlicesForHull.GetOutput(),filename)
#    topOfHull=appendedSlicesForHull.GetOutput()
    

    bigHull=topOfHull
    #topOfHull=ta.TrimConvexHull2(esch.GetOutput(), -1,1,0.5,1)
    
    filename=dirToTest+"\\"+ "trim"  + ".vtp"
    ta.LogVTK(topOfHull,filename)
    

    
    
    filename=dirToTest+"\\"+ "bigtrim"  + ".vtp"
    ta.LogVTK(bigHull,filename)
    
    
    topOfHull=chPD
    
    croppedToGrooveShouldTopOfHulla=ta.ThresholdPointOrCellData(topOfHull,False,"yn",-0.9,0.9)
    # change 0.9 to 0.86
    croppedToGrooveShouldTopOfHull=ta.ThresholdPointOrCellData(croppedToGrooveShouldTopOfHulla,False,"zn",0.86,1)
 
#
    filename=dirToTest+"\\"+ "croppedToGrooveShouldTopOfHull.vtp"
    ta.LogVTK(croppedToGrooveShouldTopOfHull,filename)

    
    bdsHull=croppedToGrooveShouldTopOfHull.GetBounds()
    bds=clippedRawTireInsideTargetBarPD.GetBounds()
    clippedToTopOfHullClippedRawTireInsideTargetBarPD=ta.ClipPD1(clippedRawTireInsideTargetBarPD,(bdsHull[0],bdsHull[1],     bdsHull[2], bdsHull[3] ,    bds[4],bds[5]),True )  
    filename=dirToTest+"\\"+"clippedToTopOfHullClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(clippedToTopOfHullClippedRawTireInsideTargetBarPD,filename)

    #clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD=orientedClippedRawTireInsideTargetBarPD
    treadDepth = ta.ComputeDistanceNew(0,clippedToTopOfHullClippedRawTireInsideTargetBarPD,croppedToGrooveShouldTopOfHull )

    
    filename=dirToTest+"\\"+ "td1.vtp"
    ta.LogVTK(treadDepth.GetOutput(),filename)
    
    NormalizeDistanceArrayBasedOnKnownScaleTreadWidth(treadDepth.GetOutput(),tw)
    
    treadDepthWith32nds =Add32ndsToSwath(treadDepth.GetOutput())
    filename=dirToTest +"\\"+ "td32nds.vtp"
    ta.LogVTK(treadDepthWith32nds,filename)
 
    tireSwath=treadDepthWith32nds
    tireSwathBds=tireSwath.GetBounds()
    
 
    

#    
    blockSwathDecPD,grooveSwathDecPD=ModifySwathForDisplay(tireSwath)
    


    
    midTireSlicePD,midProfileLine=ReturnMiddleProfile(tireSwath)
    midTireSliceActor=ReturnProfileActor(midProfileLine)
    

    
    rotatedMidTireSlicePD=GenerateTransformedSlice(midTireSlicePD,tireSwath) 
    rotatedMidTireSliceActor=ReturnProfileActor(rotatedMidTireSlicePD)
    

    
#    
#    filename=dirToTest+"\\"+ "act_midTireSlicePD.vtp"
#    ta.LogVTK(midTireSlicePD,filename)
#    filename=dirToTest+"\\"+ "act_rotatedMidTireSlicePD.vtp"
#    ta.LogVTK(rotatedMidTireSlicePD,filename)
#    
    
    #
    deepGroovePointList=ReturnMidTireProfileInformation(rotatedMidTireSlicePD)
    lowerLUT=int( round(min(deepGroovePointList[:,3] -0.5) ) )
    upperLUT= int( round(max(deepGroovePointList[:,3] +0.5) ) )
      
    
    lut=GenLUT(lowerLUT,upperLUT)
    lutRange=(lowerLUT,upperLUT)
    
    
    
    #
    textActorList=GenerateTextAnnotationForGrooves(tireSwathBds[3]+0.001, deepGroovePointList)

        

   
    #
    blockSwathDecActor,grooveSwathDecActor=PrepareBlocksAndGroovesForRendering(blockSwathDecPD,grooveSwathDecPD,lut,lutRange)

    
        
    #scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.2), 0.34,lut)
    scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.1), 0.5,lut)
    

    
#    metaDataFile= dirToTest+"\\"+ commonSettings.metaDataFile
    
    print ("\n path is ",dirToTest )

    
    
    actorList1=[]
    actorList2=[]
    actorList3=[]
    actorList=[]
    
#    actorList1.append(scalar_bar)
    
#    for item in textActorList:
#        actorList3.append ( item )
    actorList2=textActorList
    #actorList3.append ( titleActor2 )
    
   
    actorList2.append(midTireSliceActor)

    
    actorList2.append(rotatedMidTireSliceActor)
    
    actorList2.append(blockSwathDecActor)
    actorList2.append(grooveSwathDecActor)
    
    actorList2.append(scalar_bar)
       
    # top of report
    #ren1 = vtk.vtkRenderer()
    ren2 = vtk.vtkRenderer()
    #ren3 = vtk.vtkRenderer()
    
    renWin = vtk.vtkRenderWindow()
#    ren1.AddActor2D(scalar_bar)
    #RenderActors(actorList1,ren1)
    RenderActors(actorList2,ren2)
    #RenderActors(actorList3,ren3)
       
    #renWin.AddRenderer(ren1)
    renWin.AddRenderer(ren2)
    #renWin.AddRenderer(ren3)
    
#    ren1.SetViewport(0, 0, 0.1, 1)
#    ren2.SetViewport(0.1, 0.0, 1, 0.9)
#    ren3.SetViewport(0.1, 0.91, 1, 1)
    
#    ren1.SetViewport(0, 0, 1, 0.3)
#    ren2.SetViewport(0, 0.31, 1, 0.9)
#    ren3.SetViewport(0, 0.91, 1, 1)
    
    
    ren2.SetViewport(0, 0, 1, 1)
    #ren3.SetViewport(0, 0.91, 1, 1)
    
  
    renderWindowX=1000
    renderWindowY=772
    
    renWin.SetSize(renderWindowX,renderWindowY)
    #ren1.SetBackground(0.1, 0.2, 0.4)
#    ren1.SetBackground(1,1,1)
    ren2.SetBackground(1,1,1)
#    ren3.SetBackground(1,1,1)
    
    camera =vtk.vtkCamera ()
    camera.SetPosition(0, 0,0.44)
    ren2.SetActiveCamera(camera)
#    camera.SetFocalPoint(0, 0, 0)
    #ren2.GetActiveCamera().Zoom(1.5)
    
   
    renWin.Render()
     
    reportname="TireAuditReport"
  
    reportdirToTest=dirToTest+"\\"+reportname
    reportdirToTestJPG=dirToTest+"\\"+reportname+".png"
    

    
    #ExportAsPDF(reportdirToTest,renWin)
    ExportAsPNG(reportdirToTestJPG,renWin)
    #ExportAsPDF(reportdirToTest,renWin)

    
#    renWin = ren.GetRenderWindow()
    renWin.Finalize()
#    ren.TerminateApp()
    
    del renWin, ren2
    print("\n*** report path is ", reportdirToTest)
    AddLogo(reportdirToTestJPG)
    
    
    
    

    print ("\n hello")




pollTireStates.pollTireTransactionStatus(ProcessReportNew1,  tireProcessStateSettings.stateSmartPhoneOnlyReadyForReportProcessing,tireProcessStateSettings.stateSmartPhoneOnlyReadyForEmailProcessing,3)


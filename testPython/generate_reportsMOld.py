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

import databaseSettings

#databaseSettings = imp.load_source('databaseSettings', 'C:\\Projects\\tireProgramming\\testPython\\databaseSettings.py')
ta = imp.load_source('tireauditroutines', '..\\lib\\tireauditroutines.py')




def GetMetaData(dirToTest):
    
    tireIDPathOnly=dirToTest[dirToTest.rfind('\\')+1:len(dirToTest)]
    
    selectString1="SELECT  \
    Scan.ScanID,  \
    Scan.TireID,  \
    Scan.Path,  \
    Tire_copy.DOTCode,   \
    Tire_copy.Comments,  \
    Tire_copy.NewWornFlag,  \
    Tire_copy.MountedOnRim,  \
    Tire_copy.MountedOnVehicle,   \
    TireType.BrandName,  \
    TireType.SectionWidth,  \
    Manufacturer.ManufacturerName,  \
    Tire_copy.TireTypeID,  \
    TireType.PassengerCommercialFlag, \
    TireType.AspectRatio,\
    TireType.Radius \
    FROM Manufacturer , Scan INNER JOIN TireType ON Manufacturer.ManufacturerID = TireType.ManufacturerID   \
    INNER JOIN Tire_copy ON TireType.TireTypeID = Tire_copy.TireTypeID AND Tire_copy.TireID = Scan.TireID \
    WHERE Scan.Path=" 
    selectString2= "\"" + tireIDPathOnly +"\""
    selectString3="ORDER BY \
    Tire_copy.TireID ASC, \
    TireType.TireTypeID ASC, \
    Tire_copy.NewWornFlag ASC "
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
    tirefullpath=dirToTest+"\\"+tname
   
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(tirefullpath)
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
    boundsInsideTargetBar=[ boundsRawMesh[0],boundsRawMesh[1],generate_reportsMSettings.yLower+generate_reportsMSettings.heightOfTargetBar,generate_reportsMSettings.yHigher-generate_reportsMSettings.heightOfTargetBar,boundsRawMesh[5]-generate_reportsMSettings.depthOfTargetBar-generate_reportsMSettings.maxTreadDepth ,boundsRawMesh[5]]
    clippedRawTireInsideTargetBar=ta.Clip(pdLargeComponent,boundsInsideTargetBar)
    filename=dirToTest+ "\\"+"clippedRawTireInsideTargetBar.vtp"
    ta.LogVTK(clippedRawTireInsideTargetBar.GetOutput(), filename)
    
    return(clippedRawTireInsideTargetBar.GetOutput())

def Add32ndsToSwath(pd):
    
    for i in range(0,pd.GetPointData().GetNumberOfArrays()):
        if pd.GetPointData().GetArrayName(i)=="Distance":
            distIndex=i
            break
    
    distPts = pd.GetPointData().GetArray(distIndex)   
    numPts =  pd.GetNumberOfPoints()
#    vectors to 3 scalar arrays
#    
#    dataDist = vtk.vtkIntArray()
#    dataDist.SetNumberOfComponents(1)
#    dataDist.SetName("DistanceIn32nds")
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
    
    ta.LogVTK(cutterA.GetOutput(),"c:\\temp\\slice123.vtp")
    
    return(cutterA.GetOutput())
    
    

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
        aLabelTransform.Translate(deepGroovePointList[i][0],deepGroovePointList[i][1], 0)
        
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
        
        

        labelActor.GetProperty().SetColor ( 0,0,0)
  
        textActorList.append(labelActor)
    
    return(textActorList)
        
 

def GenerateTransformedSlice(pd) :
    
    rotatedsliceTransform = vtk.vtkTransform()
    #rotatedsliceTransform.Identity()
    rotatedsliceTransform.RotateX(270)
    rotatedsliceTransform.PostMultiply()
    # put back to 0.1
    rotatedsliceTransform.Translate(0,0.13, 0)

    rotatedsliceTransformPDFilter= vtk.vtkTransformPolyDataFilter()
    rotatedsliceTransformPDFilter.SetTransform(rotatedsliceTransform)
    rotatedsliceTransformPDFilter.SetInputData(pd)
    rotatedsliceTransformPDFilter.Update()
    

   # Create a mapper and actor to display the text.
    rotatedsliceMapper = vtk.vtkPolyDataMapper()
    rotatedsliceMapper.SetInputConnection(rotatedsliceTransformPDFilter.GetOutputPort())
    
    rotatedsliceActor = vtk.vtkActor()
    rotatedsliceActor.SetMapper(rotatedsliceMapper)
  


    return(rotatedsliceTransformPDFilter.GetOutput())
   

def ExportAsJPG(reportName,renWin):
        
    vtkW2Image =vtk.vtkWindowToImageFilter()
    vtkW2Image.SetInputBufferTypeToRGB()
    vtkW2Image.SetInput(renWin)
    vtkW2Image.Update()
    
    vtkJPEGWriter=vtk.vtkJPEGWriter()
    vtkJPEGWriter.SetFileName(reportName)
    
    vtkJPEGWriter.SetInputData(vtkW2Image.GetOutput())
    vtkJPEGWriter.Write()
  
    


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
    scalar_bar.SetLabelFormat("%.0f")
    scalar_bar.SetLabelTextProperty(textProp)
    scalar_bar.SetTitleTextProperty(titleTextProp)
    
 
    
    #scalar_bar.SetDrawFrame(1)
    #scalar_bar.SetWidth(.30)
    #SetVerticalTitleSeparation	(	int 		)	
    
    
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


def ClipEndTillStable(pd,clipRight,xStep):
    
    cleanHull=vtk.vtkCleanPolyData()
    cleanHull.SetInputData(pd)
    cleanHull.Update()
    pd=cleanHull.GetOutput()
    
    detectJunkThresholdSize=20

    bounds=pd.GetBounds()
    # precision problemm?
    #bounds=[bounds[0], bounds[1],-999,999,bounds[4],bounds[5]]
    startingXPos=(bounds[0]+bounds[1])/2
    i=0
    numberCC=1
    #while (numberCC==1): 
    while (True): 

      
        if (clipRight):
            revisedBounds=[bounds[0], startingXPos,bounds[2],bounds[3],bounds[4],bounds[5]]
            startingXPos=startingXPos+xStep
        else:
            revisedBounds=[startingXPos, bounds[1],bounds[2],bounds[3],bounds[4],bounds[5]]
            startingXPos=startingXPos-xStep
        
            
        
        clippedOuterTireSwath= ta.ClipPD1(pd,revisedBounds,False)
        print ("\n number of point", clippedOuterTireSwath.GetNumberOfPoints() )
        if (clippedOuterTireSwath.GetNumberOfPoints())==0:
            break
        
        #numberCC=ta.ReturnNumberOfConnectedComponents(clippedOuterTireSwath)
        (numberCC,sizeList)=ta.ReturnNumberOfConnectedComponentsAndSizes(clippedOuterTireSwath)
        if (numberCC>1):
            if (sizeList[1][1]>detectJunkThresholdSize):
                break
        
        
        i=i+1

        fileName="c:\\temp\\clipx_"+ str(i)+ ".vtp"
        ta.LogVTK(clippedOuterTireSwath,fileName)
        
    return(startingXPos-xStep)

def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()


def ProcessReportNew(dirToTest):
    
    clippedRawTireInsideTargetBarPD=ProcessRaw3DClipToFrame(dirToTest)
    # just added 1/26/2016
    clippedRawTireInsideTargetBarPD=ta.KeepOnlyLargestConnectedComponent1(clippedRawTireInsideTargetBarPD)
    
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
    
    for i in range(0,numcuts):
     
        clipPD=ta.ClipPD1(clippedRawTireInsideTargetBarPD,(bds[0],bds[1],     yCurrent, yCurrent+yInterval ,    bds[4],bds[5]),True )  
        chPD=ta.ConvexHullSciPy(clipPD)
        #trchPD=ta.TrimConvexHull(chPD)
        #trchPD= ta.TrimConvexHull2(chPD,-1,1,-0.5,0.5,0,1)
        trchPD= ta.TrimConvexHull2(chPD,-1,1,-0.9,0.9,0,1)
        
        filename=smallHullDirectory+ "\\" + "clitem_" +str(i)+"_.vtp"
        ta.LogVTK(clipPD,filename)
 
        
        filename=smallHullDirectory+ "\\" + "chitem_" +str(i)+"_.vtp"
       
        ta.LogVTK(chPD,filename)
        
        filename=smallHullDirectory+ "\\" + "trchitem_" +str(i)+"_.vtp"
    
        ta.LogVTK(trchPD,filename)
        yCurrent=yCurrent+yInterval
        
        appendedSlicesForHull.AddInputData(trchPD)
        appendedSlicesForHull.Update()
        a=0
    
    filename="c:\\temp\\appendedSlicesForHull.vtp"
    ta.LogVTK(appendedSlicesForHull.GetOutput(),filename)
    topOfHull=appendedSlicesForHull.GetOutput()
    

#OLD      
#    #ch=ta.ConvexHullSciPy(clippedRawTireInsideTargetBarPD)
#    ch=ta.DelaunayVTK(clippedRawTireInsideTargetBarPD)
#
#    
#    esch = vtk.vtkDataSetSurfaceFilter()
#    esch.SetInputData(ch)
#    esch.Update()
#    
#    filename=dirToTest+ "\\"+"esch"  + ".vtp"
#    ta.LogVTK(esch.GetOutput(),filename)
#    
#    topOfHull=ta.TrimConvexHull1(esch.GetOutput())
# OLD      
      
    # remove stray junk
    ##topOfHull=ta.KeepOnlyLargestConnectedComponent1(topOfHull)
    bigHull=topOfHull
    #topOfHull=ta.TrimConvexHull2(esch.GetOutput(), -1,1,0.5,1)
    
    filename=dirToTest+"\\"+ "trim"  + ".vtp"
    ta.LogVTK(topOfHull,filename)
    
#    bigHull=ta.TrimConvexHull2(esch.GetOutput(),-1.0,1.0,-0.8, 0.8,0,1.0)
    ##bigHull=ta.KeepOnlyLargestConnectedComponent1(bigHull)
    
    
    filename=dirToTest+"\\"+ "bigtrim"  + ".vtp"
    ta.LogVTK(bigHull,filename)
    
    
    
    #  OLD stuff 1/26/2017
    
#    xcpRight=ClipEndTillStable(topOfHull,True,0.001)
#    xcpLeft=ClipEndTillStable(topOfHull,False,0.001)
#    
#    bounds=topOfHull.GetBounds() 
#  
#    revisedBounds=[xcpLeft, xcpRight,bounds[2],bounds[3],-999,bounds[5]]
#
#
#    croppedToGrooveShouldTopOfHull= ta.ClipPD1(topOfHull,revisedBounds,True)
#    croppedToGrooveClippedRawTireInsideTargetBarPD= ta.ClipPD1(clippedRawTireInsideTargetBarPD,revisedBounds,True)
#    croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD= ta.ClipPD1(clippedRawTireInsideTargetBarPD,revisedBounds,False)

    # end of OLD stuff 1/26/2017
    # just added 1/26/2017
    croppedToGrooveShouldTopOfHull=ta.ThresholdPointOrCellData(topOfHull,False,"zn",0.9,1)
    
#    croppedToGrooveShouldTopOfHull=ta.ConvexHullSciPy(croppedToGrooveShouldTopOfHull1)
#    croppedToGrooveShouldTopOfHull=ta.TrimConvexHull2(croppedToGrooveShouldTopOfHull,-1.0,1.0,-0.8, 0.8,0,1.0)

#
    filename=dirToTest+"\\"+ "croppedToGrooveShouldTopOfHull.vtp"
    ta.LogVTK(croppedToGrooveShouldTopOfHull,filename)
#
#    filename=dirToTest+"\\"+ "croppedToGrooveClippedRawTireInsideTargetBarPD.vtp"
#    ta.LogVTK(croppedToGrooveClippedRawTireInsideTargetBarPD,filename)
#    
#    filename=dirToTest+"\\"+ "croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD.vtp"
#    ta.LogVTK(croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD,filename)
    
    bdsHull=croppedToGrooveShouldTopOfHull.GetBounds()
    bds=clippedRawTireInsideTargetBarPD.GetBounds()
    clippedToTopOfHullClippedRawTireInsideTargetBarPD=ta.ClipPD1(clippedRawTireInsideTargetBarPD,(bdsHull[0],bdsHull[1],     bdsHull[2], bdsHull[3] ,    bds[4],bds[5]),True )  
    filename=dirToTest+"\\"+"clippedToTopOfHullClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(clippedToTopOfHullClippedRawTireInsideTargetBarPD,filename)
    orientTireXYTransformBasedOnTire=ta.PCAPD(clippedToTopOfHullClippedRawTireInsideTargetBarPD)
    orientedBasedOnTireClippedToTopOfHullClippedRawTireInsideTargetBarPD=ta.TransformPD(clippedToTopOfHullClippedRawTireInsideTargetBarPD,orientTireXYTransformBasedOnTire).GetOutput()
    filename=dirToTest+"\\"+ "orientedBasedOnTireClippedToTopOfHullClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(orientedBasedOnTireClippedToTopOfHullClippedRawTireInsideTargetBarPD,filename)
#    
    
    
    
    
    
    # 
    orientTireXYTransform=ta.PCAPD(croppedToGrooveShouldTopOfHull)
    orientedCroppedToGrooveShouldTopOfHull=ta.TransformPD(croppedToGrooveShouldTopOfHull,orientTireXYTransform).GetOutput()
    orientedClippedRawTireInsideTargetBarPD=ta.TransformPD(clippedRawTireInsideTargetBarPD,orientTireXYTransform).GetOutput()
    
    
    #added
    #orientTireXYTransformClippedRawTireInsideTargetBarPD= ta.TransformPD(croppedToGrooveClippedRawTireInsideTargetBarPD,orientTireXYTransform).GetOutput()
    
    #
    filename=dirToTest+"\\"+"orientedCroppedToGrooveShouldTopOfHull.vtp"
    ta.LogVTK(orientedCroppedToGrooveShouldTopOfHull,filename)
    bdsHull=orientedCroppedToGrooveShouldTopOfHull.GetBounds()
    
    filename=dirToTest+"\\"+"orientedClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(orientedClippedRawTireInsideTargetBarPD,filename)
    bds=orientedClippedRawTireInsideTargetBarPD.GetBounds()

    clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD=ta.ClipPD1(orientedClippedRawTireInsideTargetBarPD,(bdsHull[0]+0.000,bdsHull[1]-0.0,     bdsHull[2], bdsHull[3] ,    bds[4],bds[5]),True )  
    filename=dirToTest+"\\"+"clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD,filename)
    

    #clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD=orientedClippedRawTireInsideTargetBarPD
    treadDepth = ta.ComputeDistanceNew(0,clippedToTopOfHullOrientedClippedRawTireInsideTargetBarPD,orientedCroppedToGrooveShouldTopOfHull )

    
    filename=dirToTest+"\\"+ "td1.vtp"
    ta.LogVTK(treadDepth.GetOutput(),filename)
    
    treadDepthWith32nds =Add32ndsToSwath(treadDepth.GetOutput())
    filename=dirToTest +"\\"+ "td32nds.vtp"
    ta.LogVTK(treadDepthWith32nds,filename)
 
    tireSwath=treadDepthWith32nds
    
    orientTireXYTransformTreadDepthWith32nds= ta.TransformPD(treadDepthWith32nds,orientTireXYTransform).GetOutput()
    filename=dirToTest+"\\"+"orientTireXYTransformTreadDepthWith32nds.vtp"
    ta.LogVTK(orientTireXYTransformTreadDepthWith32nds,filename)

    
#    wholeTireSourceFilename="D:\\temp\\TireAuditRoot\\TireScans\\2016-07-04_23-30-13\\td32nds.vtp"
#    wholeTireSourceFilename="D:\\temp\\TireAuditRoot\\TireScans\\2016-07-04_23-30-13\\td32nds.vtp"
#
#    
#    reader=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(wholeTireSourceFilename )
#    reader.Update()
#    tireSwath=reader.GetOutput()  
    
#    
    blockSwathDecPD,grooveSwathDecPD=ModifySwathForDisplay(tireSwath)
    

    actorList=[]
    

    
    midTireSlicePD=ReturnMiddleProfile(tireSwath)
    midTireSliceActor=ReturnProfileActor(midTireSlicePD)
    actorList.append(midTireSliceActor)
    
    rotatedMidTireSlicePD=GenerateTransformedSlice(midTireSlicePD) 
    rotatedMidTireSliceActor=ReturnProfileActor(rotatedMidTireSlicePD)
    
    actorList.append(rotatedMidTireSliceActor)
    #
    deepGroovePointList=ReturnMidTireProfileInformation(rotatedMidTireSlicePD)
    lowerLUT=int( round(min(deepGroovePointList[:,3] -0.5) ) )
    upperLUT= int( round(max(deepGroovePointList[:,3] +0.5) ) )
      
    
    lut=GenLUT(lowerLUT,upperLUT)
    lutRange=(lowerLUT,upperLUT)
    
    
    
    #
    textActorList=GenerateTextAnnotationForGrooves(0.14, deepGroovePointList)
    for item in textActorList:
        actorList.append ( item )
        

   
    #
    blockSwathDecActor,grooveSwathDecActor=PrepareBlocksAndGroovesForRendering(blockSwathDecPD,grooveSwathDecPD,lut,lutRange)
    actorList.append(blockSwathDecActor)
    actorList.append(grooveSwathDecActor)
    
        
    #scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.2), 0.34,lut)
    scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.1), 0.5,lut)
    
    actorList.append(scalar_bar)
    
#    metaDataFile= dirToTest+"\\"+ commonSettings.metaDataFile
    
    print ("\n path is ",dirToTest )
    item=GetMetaData(dirToTest)
    
    if (item==0):
        tiremodel = "Model Not Specified"
        tiresize="Size Not Specified"
        dotCode="Not Specified"
    else:        
        tiremodel=item[10] + " " + item[8]
        tiresize=str(item[9])+" " + str(item[13]) + " " +str(item[14])
        dotCode=item[3]
    
    

    
    
    
#    titleActor=GenerateText ("Goodyear Eagle RS-A (195/60/R15)", (150,925), 40,  (0,0,0) )
    titleActor=GenerateText (tiremodel + " " + tiresize, (150,925), 40,  (0,0,0) )
    actorList.append ( titleActor )
    
    titleActor2=GenerateText ("DOT Code: M6RB JX1R 0813", (260,850), 30,  (0,0,0) )
    titleActor2=GenerateText (dotCode, (260,850), 30,  (0,0,0) )

    actorList.append ( titleActor2 )
    
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    RenderActors(actorList,ren)
    
    
    renWin.AddRenderer(ren)
    renderWindowX=1000
    renderWindowY=1000
    
    renWin.SetSize(renderWindowX,renderWindowY)
    ren.SetBackground(0.1, 0.2, 0.4)
    ren.SetBackground(1,1,1)
    renWin.Render()
     
    reportname="TireAuditReportOBHull"
  
    reportfullpath=dirToTest+"\\"+reportname
    reportfullpathJPG=dirToTest+"\\"+reportname+".jpg"
    

    
    #ExportAsPDF(reportfullpath,renWin)
    ExportAsJPG(reportfullpathJPG,renWin)
    #ExportAsPDF(reportfullpath,renWin)
    
    
#    renWin = ren.GetRenderWindow()
    renWin.Finalize()
#    ren.TerminateApp()
    
    del renWin, ren
    
    

    print ("\n hello")


def ProcessReport(dirToTest):
    
    clippedRawTireInsideTargetBarPD=ProcessRaw3DClipToFrame(dirToTest)
    
      
    #ch=ta.ConvexHullSciPy(clippedRawTireInsideTargetBarPD)
    ch=ta.DelaunayVTK(clippedRawTireInsideTargetBarPD)

    
    esch = vtk.vtkDataSetSurfaceFilter()
    esch.SetInputData(ch)
    esch.Update()
    
    filename=dirToTest+ "\\"+"esch"  + ".vtp"
    ta.LogVTK(esch.GetOutput(),filename)
    
    topOfHull=ta.TrimConvexHull1(esch.GetOutput())
    # remove stray junk
    topOfHull=ta.KeepOnlyLargestConnectedComponent1(topOfHull)
    #topOfHull=ta.TrimConvexHull2(esch.GetOutput(), -1,1,0.5,1)
    
    filename=dirToTest+"\\"+ "trim"  + ".vtp"
    ta.LogVTK(topOfHull,filename)
    
    bigHull=ta.TrimConvexHull2(esch.GetOutput(),-1.0,1.0,-0.8, 0.8,0,1.0)
    bigHull=ta.KeepOnlyLargestConnectedComponent1(bigHull)
    
    
    filename=dirToTest+"\\"+ "bigtrim"  + ".vtp"
    ta.LogVTK(bigHull,filename)
    
    
    
    xcpRight=ClipEndTillStable(topOfHull,True,0.001)
    xcpLeft=ClipEndTillStable(topOfHull,False,0.001)
    
    
    bounds=topOfHull.GetBounds() 
    #revisedBounds=[xcpLeft, xcpRight,bounds[2],bounds[3],bounds[4],bounds[5]]
    revisedBounds=[xcpLeft, xcpRight,bounds[2],bounds[3],-999,bounds[5]]


    croppedToGrooveShouldTopOfHull= ta.ClipPD1(topOfHull,revisedBounds,True)
    croppedToGrooveClippedRawTireInsideTargetBarPD= ta.ClipPD1(clippedRawTireInsideTargetBarPD,revisedBounds,True)
    croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD= ta.ClipPD1(clippedRawTireInsideTargetBarPD,revisedBounds,False)



    filename=dirToTest+"\\"+ "croppedToGrooveShouldTopOfHull.vtp"
    ta.LogVTK(croppedToGrooveShouldTopOfHull,filename)

    filename=dirToTest+"\\"+ "croppedToGrooveClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(croppedToGrooveClippedRawTireInsideTargetBarPD,filename)
    
    filename=dirToTest+"\\"+ "croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD.vtp"
    ta.LogVTK(croppedToGrooveOutsideToSWClippedRawTireInsideTargetBarPD,filename)
    
    # 
    orientTireXYTransform=ta.PCAPD(croppedToGrooveClippedRawTireInsideTargetBarPD)
    
    #added
    #orientTireXYTransformClippedRawTireInsideTargetBarPD= ta.TransformPD(croppedToGrooveClippedRawTireInsideTargetBarPD,orientTireXYTransform).GetOutput()
    
    #
    filename=dirToTest+"\\"+"orientTireXYTransform.vtp"
    ta.LogVTK(ta.TransformPD(croppedToGrooveClippedRawTireInsideTargetBarPD,orientTireXYTransform).GetOutput(),filename)

    
    #treadDepthIncludeSW = ta.ComputeDistanceNew(0,orientTireXYTransformClippedRawTireInsideTargetBarPD,bigHull )
    # removed
    treadDepthIncludeSW = ta.ComputeDistanceNew(0,clippedRawTireInsideTargetBarPD,bigHull )
    filename=dirToTest+"\\"+ "tdincludesw.vtp"
    ta.LogVTK(treadDepthIncludeSW.GetOutput(),filename)


    treadDepth = ta.ComputeDistanceNew(0,croppedToGrooveClippedRawTireInsideTargetBarPD,croppedToGrooveShouldTopOfHull )
    
    filename=dirToTest+"\\"+ "td1.vtp"
    ta.LogVTK(treadDepth.GetOutput(),filename)
    
    treadDepthWith32nds =Add32ndsToSwath(treadDepth.GetOutput())
    filename=dirToTest +"\\"+ "td32nds.vtp"
    ta.LogVTK(treadDepthWith32nds,filename)
 
    tireSwath=treadDepthWith32nds
    
    orientTireXYTransformTreadDepthWith32nds= ta.TransformPD(treadDepthWith32nds,orientTireXYTransform).GetOutput()
    filename=dirToTest+"\\"+"orientTireXYTransformTreadDepthWith32nds.vtp"
    ta.LogVTK(orientTireXYTransformTreadDepthWith32nds,filename)

    
#    wholeTireSourceFilename="D:\\temp\\TireAuditRoot\\TireScans\\2016-07-04_23-30-13\\td32nds.vtp"
#    wholeTireSourceFilename="D:\\temp\\TireAuditRoot\\TireScans\\2016-07-04_23-30-13\\td32nds.vtp"
#
#    
#    reader=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(wholeTireSourceFilename )
#    reader.Update()
#    tireSwath=reader.GetOutput()  
    
#    
    blockSwathDecPD,grooveSwathDecPD=ModifySwathForDisplay(tireSwath)
    

    actorList=[]
    

    
    midTireSlicePD=ReturnMiddleProfile(tireSwath)
    midTireSliceActor=ReturnProfileActor(midTireSlicePD)
    actorList.append(midTireSliceActor)
    
    rotatedMidTireSlicePD=GenerateTransformedSlice(midTireSlicePD) 
    rotatedMidTireSliceActor=ReturnProfileActor(rotatedMidTireSlicePD)
    
    actorList.append(rotatedMidTireSliceActor)
    #
    deepGroovePointList=ReturnMidTireProfileInformation(rotatedMidTireSlicePD)
    lowerLUT=int( round(min(deepGroovePointList[:,3] -0.5) ) )
    upperLUT= int( round(max(deepGroovePointList[:,3] +0.5) ) )
      
    
    lut=GenLUT(lowerLUT,upperLUT)
    lutRange=(lowerLUT,upperLUT)
    
    
    
    #
    textActorList=GenerateTextAnnotationForGrooves(0.14, deepGroovePointList)
    for item in textActorList:
        actorList.append ( item )
        

   
    #
    blockSwathDecActor,grooveSwathDecActor=PrepareBlocksAndGroovesForRendering(blockSwathDecPD,grooveSwathDecPD,lut,lutRange)
    actorList.append(blockSwathDecActor)
    actorList.append(grooveSwathDecActor)
    
        
    #scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.2), 0.34,lut)
    scalar_bar=GenerateLegend(upperLUT-lowerLUT+1, "Groove Depth (32nds) ", (.1,0.05) , (0.8,0.1), 0.5,lut)
    
    actorList.append(scalar_bar)
    
#    metaDataFile= dirToTest+"\\"+ commonSettings.metaDataFile
    
    print ("\n path is ",dirToTest )
    item=GetMetaData(dirToTest)
    
    if (item==0):
        tiremodel = "Model Not Specified"
        tiresize="Size Not Specified"
        dotCode = "DOT Not Spec"
    else:    
        tiremodel=item[10] + " " + item[8]
        tiresize=str(item[9])+" " + str(item[13]) + " " +str(item[14])
        dotCode=item[3]
        

    
    
    
    
#    titleActor=GenerateText ("Goodyear Eagle RS-A (195/60/R15)", (150,925), 40,  (0,0,0) )
    titleActor=GenerateText (tiremodel + " " + tiresize, (150,925), 40,  (0,0,0) )
    actorList.append ( titleActor )
    
    titleActor2=GenerateText ("DOT Code: M6RB JX1R 0813", (260,850), 30,  (0,0,0) )
    titleActor2=GenerateText (dotCode, (260,850), 30,  (0,0,0) )

    actorList.append ( titleActor2 )
    
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    RenderActors(actorList,ren)
    
    
    renWin.AddRenderer(ren)
    renderWindowX=1000
    renderWindowY=1000
    
    renWin.SetSize(renderWindowX,renderWindowY)
    ren.SetBackground(0.1, 0.2, 0.4)
    ren.SetBackground(1,1,1)
    renWin.Render()
     
    reportname="TireAuditReport"
  
    reportfullpath=dirToTest+"\\"+reportname
    reportfullpathJPG=dirToTest+"\\"+reportname+".jpg"
    

    
   
    ExportAsJPG(reportfullpathJPG,renWin)
    #ExportAsPDF(reportfullpath,renWin)
    
    
#    renWin = ren.GetRenderWindow()
    renWin.Finalize()
#    ren.TerminateApp()
    
    del renWin, ren
    
    

    print ("\n hello")




pollTireStates.pollTireTransactionStatus(ProcessReportNew,  tireProcessStateSettings.stateReadyForReportProcessing,tireProcessStateSettings.stateReadyForEmailProcessing,3)


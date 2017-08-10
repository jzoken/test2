# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 21:42:34 2015

@author: jzoken
"""

import vtk
import csv
import numpy as np
from matplotlib import pyplot as plt
import sys
import random 
import os
#import plyfile as pf
#import transformations as tr
import math
#import main1
import imp

import time
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from vtk.util import numpy_support
import pcl

import paramiko


pf = imp.load_source('plyfile', 'C:\\Projects\\tireProgramming\\lib\\plyfile.py')
tr = imp.load_source('transformations', 'C:\\Projects\\tireProgramming\\lib\\transformations.py')



#global directoryForLog
directoryForLog="c:\\temp\\tireaudittesting\\"

def TranslateTireSwathToOriginXY(tireSwath):
    
    bounds=[0,0,0,0,0,0]
    bounds=tireSwath.GetBounds()
    
    xTranslate = -(bounds[1]-bounds[0])/2
    yTranslate =-(bounds[3]-bounds[2])/2
    
        
    orginTransform=vtk.vtkTransform()
    orginTransform.Translate( xTranslate,yTranslate,0)
    
    orientedTireSwath=TransformPD(tireSwath,orginTransform).GetOutput()
      
    return(orientedTireSwath)

def DelaunayVTK(pd):
    
    ch = vtk.vtkDelaunay3D()
    ch.SetInputData(pd)
    ch.Update()
     
    esch = vtk.vtkDataSetSurfaceFilter()
    esch.SetInputConnection(ch.GetOutputPort())
    esch.Update()
    
    return(esch.GetOutput())

def ConvexHullSciPy(pd):

    
    xyz_vtk_array= pd.GetPoints().GetData()
    
    xyz_numpy_array = numpy_support.vtk_to_numpy(xyz_vtk_array)
    

    hull = ConvexHull(xyz_numpy_array)
    #hull= Delaunay(xyz_numpy_array,"Qt")
    hullCells=hull.simplices
    
    hullVertices=np.unique(hullCells.flatten())
    
    vtkToNumpyVertex={}
    
    for i1 in  range(len(hullVertices)):
  
        vtkToNumpyVertex[hullVertices[i1]]=i1
      
    
    hullPts=hull.points[hull.vertices]
    
    pts=vtk.vtkPoints()
    for i1 in range(len(hullPts)):
      pts.InsertPoint(i1, hullPts[i1][0],hullPts[i1][1],hullPts[i1][2])
      print ("\n vertex", i1,hullPts[i1][0],hullPts[i1][1],hullPts[i1][2] )
    
    xyz_vtk_points=pts

    
    
    triangles = vtk.vtkCellArray()
    
    for i1 in range(len(hullCells)):
        triangle = vtk.vtkTriangle()
        print ("\n triangle", i1,vtkToNumpyVertex[hullCells[i1][0]],vtkToNumpyVertex[hullCells[i1][1]],vtkToNumpyVertex[hullCells[i1][2]],"\n" )
        triangle.GetPointIds().SetId(0,vtkToNumpyVertex[hullCells[i1][0]])
        triangle.GetPointIds().SetId(1,vtkToNumpyVertex[hullCells[i1][1]])
        triangle.GetPointIds().SetId(2,vtkToNumpyVertex[hullCells[i1][2]])
        triangles.InsertNextCell(triangle)
        
     
    pdch = vtk.vtkPolyData()
    
    pdch.SetPolys(triangles)
    pdch.SetPoints(xyz_vtk_points)
#    
#    qvertices=hull.points[hull.vertices]
#    indexVertexList=np.unique(hull.simplices)
#    hullsimpflatten=hull.simplices.flatten()
#    qfaces=np.searchsorted(indexVertexList,hullsimpflatten).reshape(-1,3)
#    
#    
#
#    #norm = np.zeros( hull.points[hull.vertices].shape, dtype=hull.points[hull.vertices].dtype )
#    norm = np.zeros( qvertices.shape, dtype=qvertices.dtype )
#    tris=hull.points[hull.simplices]
#    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
#    normalize_v3(n)
#    norm[ qfaces[:,0] ] += n
#    norm[ qfaces[:,1] ] += n
#    norm[ qfaces[:,2] ] += n
#    normalize_v3(norm)
#    
    
    #ta.LogVTK(pd, "c:\\temp\\out.vtp")
    
    return(pdch)





def TrimConvexHull(pd):
    
    # process the clipped tread


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputData(pd)
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()


    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    

     
#    thresholdy = vtk.vtkThreshold()
#    thresholdy.SetInputData(normalsch.GetOutput())
#   
#    thresholdy.ThresholdBetween(-0.4,0.4)
#    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdy.Update()
#    
#    thresholdx = vtk.vtkThreshold()
#    thresholdx.SetInputData(thresholdy.GetOutput())
#   
#    #thresholdx.ThresholdBetween(-0.4,0.4)
#    thresholdx.ThresholdBetween(-1.0,1.0)
#    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
#    thresholdx.Update()
##    
#    thresholdz = vtk.vtkThreshold()
#    thresholdz.SetInputData(thresholdx.GetOutput())
#    #thresholdx.SetAttributeModeToUseCellData()
#    #thresholdx.ThresholdBetween(-0.95,0.95)
#    thresholdz.ThresholdBetween(0,1)
#    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
#    thresholdz.Update()
    
         
    thresholdy = vtk.vtkThreshold()
    thresholdy.SetInputData(normalsch.GetOutput())
   
    thresholdy.ThresholdBetween(-0.4,0.4)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.Update()
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(thresholdy.GetOutput())
   
    #thresholdx.ThresholdBetween(-0.4,0.4)
    thresholdx.ThresholdBetween(-1.0,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
#    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdx.GetOutput())
    #thresholdx.SetAttributeModeToUseCellData()
    #thresholdx.ThresholdBetween(-0.95,0.95)
    thresholdz.ThresholdBetween(0,1)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    

    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdz.GetOutputPort())
    esz2.Update()
    
    LogVTK( esz2.GetOutput(), "c:\\temp\\trimmedhull.vtp")
    
    LogVTK( normalsch.GetOutput(), "c:\\temp\\normals1.vtp")
    print ("\n*************************\n")
  
  

#    
    return(esz2.GetOutput())
    #return(normalsch)
    
def TrimConvexHull1(pd):
    
    # process the clipped tread


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputData(pd)
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()


    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(normalsch.GetOutput())
   
    
    thresholdx.ThresholdBetween(-0.5,0.5)
    #thresholdx.ThresholdBetween(-1.0,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
#    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdx.GetOutput())
    #thresholdx.SetAttributeModeToUseCellData()
 
    thresholdz.ThresholdBetween(0.5,1)
    #thresholdz.ThresholdBetween(0.9,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    
    

    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdz.GetOutputPort())
    esz2.Update()
    
    
    # sort of a hack - the top part of the tire is deeply convex and so has more cells
    if (esz2.GetOutput().GetNumberOfCells()>30):
        # need to invert normals
        normalsch.FlipNormalsOn()
    
    
    LogVTK( esz2.GetOutput(), "c:\\temp\\trimmedhull.vtp")
  
  

#    
    return(esz2.GetOutput())
    #return(normalsch)
    

def TrimConvexHull2(pd,xt1,xt2,yt1, yt2, zt1,zt2):
    
    # process the clipped tread


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputData(pd)
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()


    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(normalsch.GetOutput())
     
    thresholdx.ThresholdBetween(xt1,xt2)
    #thresholdx.ThresholdBetween(-1.0,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
    
    thresholdy = vtk.vtkThreshold()
    thresholdy.SetInputData(thresholdx.GetOutput())
     
    thresholdy.ThresholdBetween(yt1,yt2)
    #thresholdx.ThresholdBetween(-1.0,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.Update()
    
  
#    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdy.GetOutput())
    #thresholdx.SetAttributeModeToUseCellData()
 
    thresholdz.ThresholdBetween(zt1,zt2)
    #thresholdz.ThresholdBetween(0.9,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    
    

    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdz.GetOutputPort())
    esz2.Update()
    
    
    # sort of a hack - the top part of the tire is deeply convex and so has more cells
    if (esz2.GetOutput().GetNumberOfCells()>30):
        # need to invert normals
        normalsch.FlipNormalsOn()
    
    
    LogVTK( esz2.GetOutput(), "c:\\temp\\trimmedhull.vtp")
  
  

#    
    return(esz2.GetOutput())
    #return(normalsch)
    
    
def Place3DText(pd,str,x,y,z,txtScale):

    vecText=vtk.vtkVectorText()
    vecText.SetText(str)
    vecText.Update()
    
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection( vecText.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion();
    extrude.SetVector(0,0,0)
    extrude.SetScaleFactor (1)
    extrude.Update()
    # 
    
    triangleFilter =vtk.vtkTriangleFilter()
    triangleFilter.SetInputConnection(extrude.GetOutputPort())
    triangleFilter.Update()
    
    
    tr1=vtk.vtkTransform()
    ##tr1.RotateY(180)
    
    tr2=vtk.vtkTransform()
    tr2.Translate(float(x),float(y),float(z))
    tr2.Scale(float(txtScale),float(txtScale),float(txtScale))
    
    
    tr2=vtk.vtkTransform()
    tr2.Translate(float(x),float(y),float(z))
    tr2.Scale(float(txtScale),float(txtScale),float(txtScale))
    
    textTrPD=TransformPD(TransformPD(triangleFilter.GetOutput(),tr1).GetOutput(),tr2)
    
    if (pd):
        #place text in source PD
        appendPD=vtk.vtkAppendPolyData()
        appendPD.AddInputData(textTrPD.GetOutput())
        appendPD.AddInputData(pd)
        appendPD.Update()
    
        return(appendPD.GetOutput())
    else:
        return(textTrPD)

#def Place3DText(pd,str,x,y,z,txtScale):
#
#    vecText=vtk.vtkVectorText()
#    vecText.SetText(str)
#    vecText.Update()
#    
#    extrude = vtk.vtkLinearExtrusionFilter()
#    extrude.SetInputConnection( vecText.GetOutputPort())
#    extrude.SetExtrusionTypeToNormalExtrusion();
#    extrude.SetVector(0,0,0)
#    extrude.SetScaleFactor (1)
#    extrude.Update()
#    # 
#    
#    triangleFilter =vtk.vtkTriangleFilter()
#    triangleFilter.SetInputConnection(extrude.GetOutputPort())
#    triangleFilter.Update()
#    
#    moveX=vtk.vtkTransform()
#    moveX.Translate(float(x),float(y),float(z))
#    moveX.Scale(float(txtScale),float(txtScale),float(txtScale))
#    
#    textTrPD=TransformPD(triangleFilter.GetOutput(),moveX)
#    
#    if (pd):
#        #place text in source PD
#        appendPD=vtk.vtkAppendPolyData()
#        appendPD.AddInputData(textTrPD.GetOutput())
#        appendPD.AddInputData(pd)
#        appendPD.Update()
#    
#        return(appendPD.GetOutput())
#    else:
#        return(textTrPD)



def remoteCommand(sourceDestinationCopyList, destinationSourceCopyList, command):
    

    host = '192.168.1.11'
    i = 1
    
    localpath="c:\\temp\\abh.txt"
    remotepath="/tmp/abh.txt"
    #
    # Try to connect to the host.
    # Retry a few times if it fails.
    #
    while True:
        print "Trying to connect to %s (%i/30)" % (host, i)
    
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host,username="jzoken", password="yaelyael")
            print "Connected to %s" % host
            #scp = paramiko.SCPClient(ssh.get_transport())
    #        sftp = ssh.open_sftp()
    #        sftp.put(localpath, remotepath)
    #        sftp.close()
    #        ssh.close()
            break
        except paramiko.AuthenticationException:
            print "Authentication failed when connecting to %s" % host
            sys.exit(1)
        except:
            print "Could not SSH to %s, waiting for it to start" % host
            i += 1
            time.sleep(2)
    
        # If we could not connect within time limit
        if i == 30:
            print "Could not connect to %s. Giving up" % host
            sys.exit(1)
            
    
    sftp = ssh.open_sftp()
    for sourcDestTuple in sourceDestinationCopyList:  
        sftp.put(sourcDestTuple[0], sourcDestTuple[1])
    sftp.close()
    
    # Send the command (non-blocking)
    #stdin, stdout, stderr = ssh.exec_command("my_long_command --arg 1 --arg 2")
    stdin, stdout, stderr = ssh.exec_command(command)
    time.sleep(140)
    
    
    
    # Wait for the command to terminate
#    while not stdout.channel.exit_status_ready():
#        # Only print data if there is data to read in the channel
#        if stdout.channel.recv_ready():
#            rl, wl, xl = select.select([stdout.channel], [], [], 0.0)
#            if len(rl) > 0:
#                # Print data from stdout
#                print stdout.channel.recv(1024),
    
    sftp = ssh.open_sftp()
    for destSourceTuple in destinationSourceCopyList:  
        print destSourceTuple
        sftp.get(destSourceTuple[0], destSourceTuple[1])
    sftp.close()
    
    ssh.close()


def ICPSourceTarget(sourcePD,targetPD):
    
    
    icp = vtk.vtkIterativeClosestPointTransform()
    
    icp.SetSource(sourcePD)
    icp.SetTarget(targetPD)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToRMS();
    #icp.DebugOn()
    icp.SetMaximumMeanDistance      ( 0.001 )
    icp.SetMaximumNumberOfIterations(20000)
    
    #icp.SetMaximumNumberOfIterations(4)
    #icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    
    print icp.GetMeanDistance()
    
    print icp.GetLandmarkTransform()
    
    print icp.GetLandmarkTransform().GetMatrix()
    
    return(icp)


def TransformByHalf(icpMatrix, rotatedMesh):

    

    m1=vtk.vtkMatrix4x4()

    #m1=icp.GetLandmarkTransform().GetMatrix()
    #m1=icp.GetLandmarkTransform().GetMatrix()
    m1=icpMatrix
    
    
    
    #m0.Invert(m1,m2)
    print "\n *************m1", m1
    
    
    print "\n******* m1 is ", m1
    
    
    m1np_rot=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            m1np_rot[i,j] = m1.GetElement(i,j)
            
    #m1np=[ [0.999973, 0.0018064, -0.00713717, -0.0014807 ],
    #       [ -0.0017505, 0.999968, 0.00783144, 0.000210273 ],
    #       [ 0.00715109, -0.00781874, 0.999944 ,-4.48776e-007 ],
    #        [0 ,0 ,0 ,1 ]]
        
    
    m1np_trans_dv=np.zeros((3))
    for i in range(3):
            #print "\n **&& ", m1.GetElement(4,i)
            print "\nM1 IS ", m1
            m1np_trans_dv[i] = m1.GetElement(i,3)    
    
    print "\n******* m1np_trans_dv is ", m1np_trans_dv
    
    m1np_trans_matrix=tr.translation_matrix(m1np_trans_dv)
    #
    ##m2=icp.GetLandmarkTransform().Inverse().GetMatrix()   
    #icpInvTransform =vtk.vtkTransform()  
    #icpInvTransform.SetMatrix(m2)
    
    
    a,b,c=tr.euler_from_matrix(m1np_rot)
    print "\n******** a b c ", a,b,c
    ahalf=a/2
    bhalf=b/2
    chalf=c/2
    trsl=tr.translation_from_matrix(m1np_trans_matrix)
    print "\n******* trans is ", trsl
    trslhalf=trsl/2
    m1npHalf=tr.compose_matrix(scale=None, shear=None,angles=[ahalf,bhalf,chalf],translate=trslhalf)
    
    # convert m1half back to vtk matrix and transform
    
    m1Half=vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            element=m1npHalf[i,j]
            m1Half.SetElement(i,j,element)
            print "\n i i element", i,j,element
            
    
    print "\n *************m1Half", m1Half
         
    
    # transform mesh according to m2Half      
    #dontcare,avgdist=ta.ComputeDistance(7777,new_mesh_tire,ta.TransformPD(meshImproveRotate180.GetOutput(), icp),0 , True )
    halfTransformBack=vtk.vtkTransform()
    halfTransformBack.SetMatrix(m1Half)
    meshImproveRotate180HalfBack = TransformPD(rotatedMesh, halfTransformBack)
    LogVTK(meshImproveRotate180HalfBack.GetOutput(), "c:\\temp\\rotatedGTaftericpHalf.vtp")
    
    return(meshImproveRotate180HalfBack.GetOutput(), halfTransformBack)



def GenerateMeshlabScripts1(filename,targetFaceNum, qualityThr,filterScale, curvatureType ):
   
    fo = open(filename,"wb")
    
    lines=[]
    lines.append("<!DOCTYPE FilterScript>")
    lines.append("<FilterScript>")
    
    print ("\n target******************** face", targetFaceNum)

    # 0 means don't decimate
    if (targetFaceNum<>0):
   
        defaultTargetFaceNum=5000
        defaultTargetPerc=0
        defaultQualityThr=0.05
        defaultPreserveBoundary="false"
        defaultBoundaryWeight=1
        defaultPreserveNormal="false"
        defaultPreserveTopology="false"
        defaultOptimalPlacement="true"
        defaultPlanarQuadric="false"
        defaultQualityWeight="false"
        defaultAutoClean="true"
        defaultSelected="false"
        
        if defaultTargetFaceNum==0:
            txt_targetFaceNum=str(defaultTargetFaceNum)
        else:
            txt_targetFaceNum=str(targetFaceNum)
            
        txt_targetPerc=str(defaultTargetPerc)
        
        if defaultQualityThr==0:
            txt_qualityThr=str(defaultQualityThr)
        else:
            txt_qualityThr=str(qualityThr)
        
        txt_preserveBoundary=defaultPreserveBoundary
        txt_boundaryWeight=str(defaultBoundaryWeight)
        txt_preserveNormal=defaultPreserveNormal
        txt_preserveTopology=defaultPreserveTopology
        txt_optimalPlacement=defaultOptimalPlacement
        txt_planarQuadric=defaultPlanarQuadric
        txt_qualityWeight=defaultQualityWeight
        txt_autoClean=defaultAutoClean
        txt_selected=defaultSelected
        
        
        
        

        lines.append(" <filter name=\"Quadric Edge Collapse Decimation\">")
        lines.append("  <Param type=\"RichInt\" value=\""     +txt_targetFaceNum +  "\" name=\"TargetFaceNum\"/>")
        lines.append("  <Param type=\"RichFloat\" value=\""    +txt_targetPerc+  "\" name=\"TargetPerc\"/>")
        lines.append("  <Param type=\"RichFloat\" value=\""    +txt_qualityThr + "\" name=\"QualityThr\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""      +txt_preserveBoundary+  "\" name=\"PreserveBoundary\"/>")
        lines.append("  <Param type=\"RichFloat\" value=\"" + txt_boundaryWeight+  "\" name=\"BoundaryWeight\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""     +txt_preserveNormal +  "\" name=\"PreserveNormal\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""    +txt_preserveTopology+  "\" name=\"PreserveTopology\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""    +txt_optimalPlacement + "\" name=\"OptimalPlacement\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""      +txt_planarQuadric+  "\" name=\"PlanarQuadric\"/>")
        lines.append("  <Param type=\"RichBool\" value=\"" + txt_qualityWeight+  "\" name=\"QualityWeight\"/>")
        lines.append("  <Param type=\"RichBool\" value=\""      +txt_autoClean+  "\" name=\"AutoClean\"/>")
        lines.append("  <Param type=\"RichBool\" value=\"" + txt_selected+  "\" name=\"Selected\"/>")
        lines.append(" </filter>")
        
    
    defaultSelectionOnly="false"
    defaultFilterScale=1
    defaultProjectionAccuracy=0.0001 
    defaultMaxProjectionIters=15
    defaultSphericalParameter=1
    defaultCurvatureType=2
    
    txt_selectionOnly=defaultSelectionOnly
 
    if filterScale==0:
        txt_filterScale=str(defaultFilterScale)
    else:
        txt_filterScale=str(filterScale)
        
   
    txt_ProjectionAccuracy=str(defaultProjectionAccuracy)

    txt_maxProjectionIters=str(defaultMaxProjectionIters)

    txt_sphericalParameter=str(defaultSphericalParameter)

    txt_curvatureType=str(curvatureType)
    
#    if curvatureType==0:
#        txt_curvatureType=str(defaultCurvatureType)
#        txt_filterScale=str(defaultFilterScale)
#    else:
#        txt_filterScale=str(filterScale)
#        
#    txt_curvatureType=str(curvatureType)
 
        
   
    lines.append(" <filter name=\"Colorize curvature (APSS)\">")
    lines.append("  <Param type=\"RichBool\" value=\""     +txt_selectionOnly +  "\" name=\"SelectionOnly\"/>")
    lines.append("  <Param type=\"RichFloat\" value=\""    +txt_filterScale+  "\" name=\"FilterScale\"/>")
    lines.append("  <Param type=\"RichFloat\" value=\""    +txt_ProjectionAccuracy + "\" name=\"ProjectionAccuracy\"/>")
    lines.append("  <Param type=\"RichInt\" value=\""      +txt_maxProjectionIters+  "\" name=\"MaxProjectionIters\"/>")
    lines.append("  <Param type=\"RichFloat\" value=\"" + txt_sphericalParameter+  "\" name=\"SphericalParameter\"/>")
    lineA="  <Param enum_val0=\"Mean\" enum_val1=\"Gauss\" enum_cardinality=\"5\" enum_val2=\"K1\" enum_val3=\"K2\" type=\"RichEnum\" value=\""
    lineB="\" enum_val4=\"ApproxMean\" name=\"CurvatureType\"/>"

    lines.append(lineA+txt_curvatureType+lineB)
    
    lines.append(" </filter>")
    lines.append("</FilterScript>")
    
    #print "\n", line1,"\n", line2,"\n",line3,"\n", line4, "\n", line5,"\n", line6,"\n",line7,"\n", line8,"\n", line9,"\n", line10,"\n",line11,"\n"
    for lineItem in lines:
        fo.write(lineItem+"\n")
 
    fo.close()
    
    #line4="  <Param type=\"RichBool\" value=\""+ txt_selectionOnly  + "\" name=\"SelectionOnly\"/>"
    #print "\n line 4", line4
    
#    line1="<!DOCTYPE FilterScript>"
#    line2="<FilterScript>"
#    line3=" <filter name=\"Colorize curvature (APSS)\">"
#    line4="  <Param type=\"RichBool\" value=\"%s\" name=\"SelectionOnly\"/>",  txt_selectionOnly
#    line5="  <Param type=\"RichFloat\" value=\"%f\" name=\"FilterScale\"/>", txt_filterScale
#    line6="  <Param type=\"RichFloat\" value=\"%f\" name=\"ProjectionAccuracy\"/>", txt_ProjectionAccuracy
#    line7="  <Param type=\"RichInt\" value=\"%d\" name=\"MaxProjectionIters\"/>",txt_maxProjectionIters
#    line8="  <Param type=\"RichFloat\" value=\"%f\" name=\"SphericalParameter\"/>", txt_sphericalParameter
#    line9="  <Param enum_val0=\"Mean\" enum_val1=\"Gauss\" enum_cardinality=\"5\" enum_val2=\"K1\" enum_val3=\"K2\" type=\"RichEnum\" value=\"3\" enum_val4=\"ApproxMean\" name=\"CurvatureType\"/>"
#    line10=" </filter>"
#    line11="</FilterScript>"
    
    


def ComputeCurvature1(pd, targetFaceNum, qualityThr,filterScale, curvatureType):
#    

    meshLabScriptFileName="c:\\temp\\mlscript.mlx"
    GenerateMeshlabScripts1(meshLabScriptFileName,targetFaceNum, qualityThr,filterScale, curvatureType )
    print "\n Entering ComputeMaxCurvature\n"
    dirTemp="c:\\temp\\"
    fileTemp=dirTemp+"maxcurvtemp.ply"
    fileTempCurv=dirTemp+"maxcurvtemp_curv.ply"
    fileTempCurvAscii=dirTemp+"maxcurvtemp_curv_ascii.ply"
    LogPLY(pd,fileTemp)

    cmd="C:\\\"Program Files\"\\VCG\\MeshLab\\meshlabserver"
    cmdline="%s -i %s -o %s -s %s -om vc vq  fq m" %(cmd,  fileTemp, fileTempCurv, meshLabScriptFileName)
    print "\n command line is ", cmdline
    #os.system("C:\\\"Program Files\"\\VCG\\MeshLab\\meshlabserver -i fileTemp -o c:\\temp\\maxcurvtemp_curv.ply -s c:\\temp\\maxcurv.mlx -om vc vq  fq m")
    os.system(cmdline)
       
   
    plydata = pf.PlyData.read(fileTempCurv)
    pf.PlyData(plydata, text=True).write(fileTempCurvAscii)
#    
#
#    filename="C:\\Temp\\m1h.ply_7235392_GY_279_Copy\\origmesh_oriented_pca_icp_partialclip_with_min_curv.ply"
#    filename="c:\\temp\\curvfilein.ply"
#    filename ="c:\\temp\\gr_new_curvmax.ply"
#    filename="c:\\temp\\hankook_new_curvmax.ply"
#    outfilename="c:\\temp\\gr_new_curvmax.vtp"
#    outfilename="c:\\temp\\hankook_new_curvmax.vtp"
#    filename="c:\\temp\\hankook_new_DH01_curvmax.ply"
#    outfilename="c:\\temp\\hankook_new_DH01_curvmax.vtp"
#    filename="c:\\temp\\goodrich_worn_curvmax.ply"
#    outfilename="c:\\temp\\goodrich_worn_curvmax.vtp"max*
#    script.
#    #filename="c:\\temp\\goodrich_new_curvmax.ply"
#    #outfilename="c:\\temp\\goodrich_new_curvmax.vtp"
#    #filename="c:\\temp\\ecopia_new_curvmax.ply"
#    #outfilename="c:\\temp\\ecopia_new_curvmax.vtp"
    
    filename="c:\\temp\\some_ascii.ply"
    outfilename="c:\\temp\\some_ascii.vtp"
    
    
    
    plyFileForAttributes= vtk.vtkPLYReader()
    plyFileForAttributes.SetFileName(fileTempCurvAscii)
    plyFileForAttributes.Update()
        
        
    numPoints=  plyFileForAttributes.GetOutput().GetNumberOfPoints()
    
    numberOfPtArrays=plyFileForAttributes.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Point Arrays\n", numberOfPtArrays
    for i in range(0,numberOfPtArrays):
        print  plyFileForAttributes.GetOutput().GetPointData().GetArray(i).GetName()
    
    fileStateBeforePointPropertyMetaData=0
    fileStatePointPropertyMetaData=1
    fileStateFacePropertyMetaData=2
    fileStatePointPropertyData=3
    
    fileState=fileStateBeforePointPropertyMetaData
    
    #infile=open("c:\\temp\\examplewithcurv.ply" )
    infile=open(fileTempCurvAscii)
    line = infile.readline()
    
    
    while line:
        if "element vertex" in line:
            fileState=fileStatePointPropertyMetaData
            print "\n elev vertex line\n", line, "\n*********"
            numPoints=int(line.rstrip().split(' ' )[2])
            
        else:
            if "element face" in line:
                fileState=fileStateFacePropertyMetaData
            else:
                    if "end_header" in line:
                        line = infile.readline()
                        break
        line = infile.readline()
    
    print "\n numPoints = ", numPoints
    
            
    curvatureIndex = 7
    
    
    curvatureList=[]
    
    
    for i in range(numPoints):
        #print "\n line = ", line
        attributeValue=line.split(' ')
        curvatureList.append(attributeValue[curvatureIndex])
    
        line = infile.readline()
                         
      
            
    curvature1 = vtk.vtkFloatArray()
    curvature1.SetNumberOfComponents(1)
    curvature1.SetName("curvature")
    
    
    
    
    
    
    
    
    print "\n property list"
    
        
    print "lasto", curvatureList[numPoints-1]
    
    for i,curvatureListItem in enumerate(curvatureList):
        curvature1.InsertValue(i, float(curvatureListItem))

    
    
    plyFileForAttributes.GetOutput().GetPointData().AddArray(curvature1)
    
    plyFileForAttributes.Update() 
    
    LogVTK(plyFileForAttributes.GetOutput(),"c:\\temp\\vtpwithcurv.vtp")
    
    #return(plyFileForAttributes.GetOutput())
    return(plyFileForAttributes)

def ComputeCurvature2(pd, targetFaceNum, qualityThr,filterScale, curvatureType):
#    

    meshLabScriptFileName="c:\\temp\\mlscript.mlx"
    GenerateMeshlabScripts1(meshLabScriptFileName,targetFaceNum, qualityThr,filterScale, curvatureType )
    print "\n Entering ComputeMaxCurvature\n"
    dirTemp="c:\\temp\\"
    fileTemp=dirTemp+"maxcurvtemp.ply"
    fileTempCurv=dirTemp+"maxcurvtemp_curv.ply"
    fileTempCurvAscii=dirTemp+"maxcurvtemp_curv_ascii.ply"
    LogPLY(pd,fileTemp)

    cmd="C:\\\"Program Files\"\\VCG\\MeshLab\\meshlabserver"
    cmdline="%s -i %s -o %s -s %s -om vc vq  fq m" %(cmd,  fileTemp, fileTempCurv, meshLabScriptFileName)
    print "\n command line is ", cmdline
    #os.system("C:\\\"Program Files\"\\VCG\\MeshLab\\meshlabserver -i fileTemp -o c:\\temp\\maxcurvtemp_curv.ply -s c:\\temp\\maxcurv.mlx -om vc vq  fq m")
    os.system(cmdline)
       
   
    plydata = pf.PlyData.read(fileTempCurv)
    pf.PlyData(plydata, text=True).write(fileTempCurvAscii)
#    
#
#    filename="C:\\Temp\\m1h.ply_7235392_GY_279_Copy\\origmesh_oriented_pca_icp_partialclip_with_min_curv.ply"
#    filename="c:\\temp\\curvfilein.ply"
#    filename ="c:\\temp\\gr_new_curvmax.ply"
#    filename="c:\\temp\\hankook_new_curvmax.ply"
#    outfilename="c:\\temp\\gr_new_curvmax.vtp"
#    outfilename="c:\\temp\\hankook_new_curvmax.vtp"
#    filename="c:\\temp\\hankook_new_DH01_curvmax.ply"
#    outfilename="c:\\temp\\hankook_new_DH01_curvmax.vtp"
#    filename="c:\\temp\\goodrich_worn_curvmax.ply"
#    outfilename="c:\\temp\\goodrich_worn_curvmax.vtp"max*
#    script.
#    #filename="c:\\temp\\goodrich_new_curvmax.ply"
#    #outfilename="c:\\temp\\goodrich_new_curvmax.vtp"
#    #filename="c:\\temp\\ecopia_new_curvmax.ply"
#    #outfilename="c:\\temp\\ecopia_new_curvmax.vtp"
    
    filename="c:\\temp\\some_ascii.ply"
    outfilename="c:\\temp\\some_ascii.vtp"
    
    
#    
#    plyFileForAttributes= vtk.vtkPLYReader()
#    plyFileForAttributes.SetFileName(fileTempCurvAscii)
#    plyFileForAttributes.Update()
#        
#        
#    numPoints=  plyFileForAttributes.GetOutput().GetNumberOfPoints()
#    
#    numberOfPtArrays=plyFileForAttributes.GetOutput().GetPointData().GetNumberOfArrays()
#    print "\n number of Point Arrays\n", numberOfPtArrays
#    for i in range(0,numberOfPtArrays):
#        print  plyFileForAttributes.GetOutput().GetPointData().GetArray(i).GetName()
    
    
        
        
    numPoints=  pd.GetNumberOfPoints()
    
    numberOfPtArrays=pd.GetPointData().GetNumberOfArrays()
    print "\n number of Point Arrays\n", numberOfPtArrays
    for i in range(0,numberOfPtArrays):
        print  pd.GetPointData().GetArray(i).GetName()
    
    fileStateBeforePointPropertyMetaData=0
    fileStatePointPropertyMetaData=1
    fileStateFacePropertyMetaData=2
    fileStatePointPropertyData=3
    
    fileState=fileStateBeforePointPropertyMetaData
    
    #infile=open("c:\\temp\\examplewithcurv.ply" )
    infile=open(fileTempCurvAscii)
    line = infile.readline()
    
    
    while line:
        if "element vertex" in line:
            fileState=fileStatePointPropertyMetaData
            print "\n elev vertex line\n", line, "\n*********"
            numPoints=int(line.rstrip().split(' ' )[2])
            
        else:
            if "element face" in line:
                fileState=fileStateFacePropertyMetaData
            else:
                    if "end_header" in line:
                        line = infile.readline()
                        break
        line = infile.readline()
    
    print "\n numPoints = ", numPoints
    
            
    curvatureIndex = 7
    
    
    curvatureList=[]
    
    
    for i in range(numPoints):
        #print "\n line = ", line
        attributeValue=line.split(' ')
        curvatureList.append(attributeValue[curvatureIndex])
    
        line = infile.readline()
                         
      
            
    curvature1 = vtk.vtkFloatArray()
    curvature1.SetNumberOfComponents(1)
    curvature1.SetName("curvature")
    
    
    
    
    
    
    
    
    print "\n property list"
    
        
    print "lasto", curvatureList[numPoints-1]
    
    for i,curvatureListItem in enumerate(curvatureList):
        curvature1.InsertValue(i, float(curvatureListItem))

    
    
    pd.GetPointData().AddArray(curvature1)
    

    
    
    
    LogVTK(pd,"c:\\temp\\vtpwithcurv.vtp")
    
    #return(plyFileForAttributes.GetOutput())
    return(pd)

def AddNormals(pd):
    
   

    pdNormals = vtk.vtkPolyDataNormals()
    
    pdNormals.SetInputData(pd)
    pdNormals.ComputeCellNormalsOn()
    pdNormals.SetFeatureAngle (23)
    pdNormals.SplittingOn ()
    pdNormals.Update()
    
    
    
    normCells = pdNormals.GetOutput().GetCellData().GetNormals()
    areaCells=pdNormals.GetOutput().GetCellData().GetArray(0)
    
    numCells =  pdNormals.GetOutput().GetNumberOfCells()
    #data = pdNormals.GetOutput()
    
    print "\n", "number of cells", numCells
    # convert normal vectors to 3 scalar arrays
    
    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")
    
    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")
    
    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")
    
    
    itemList=[]
    for i in range (0,numCells):
    
        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"
    
        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)
    
        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)
    
    
    itemList.sort(key=lambda tup: tup[4])
    
    
    pdNormals.GetOutput().GetCellData().AddArray(data1)
    pdNormals.GetOutput().GetCellData().AddArray(data2)
    pdNormals.GetOutput().GetCellData().AddArray(data3)
    
    #########
    
    normPoints = pdNormals.GetOutput().GetPointData().GetNormals()
    #areaCells=pdNormals.GetOutput().GetCellData().GetArray(0)
    
    numPoints =  pdNormals.GetOutput().GetNumberOfPoints()
    #data = pdNormals.GetOutput()
    
    print "\n", "number of cells", numPoints
    # convert normal vectors to 3 scalar arrays
    
    data1Pt = vtk.vtkFloatArray()
    data1Pt.SetNumberOfComponents(1)
    data1Pt.SetName("xn")
    
    data2Pt = vtk.vtkFloatArray()
    data2Pt.SetNumberOfComponents(1)
    data2Pt.SetName("yn")
    
    data3Pt = vtk.vtkFloatArray()
    data3Pt.SetNumberOfComponents(1)
    data3Pt.SetName("zn")
    
    
 
    for i in range (0,numPoints):
    
        
        ##print "\narea", area,"\n"
        n0=normPoints.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"
    
        data1Pt.InsertValue(i,x)
        data2Pt.InsertValue(i,y)
        data3Pt.InsertValue(i,z)
    
        
    
    
    pdNormals.GetOutput().GetPointData().AddArray(data1Pt)
    pdNormals.GetOutput().GetPointData().AddArray(data2Pt)
    pdNormals.GetOutput().GetPointData().AddArray(data3Pt)
    
    
    
    
    
    
    
    return(pdNormals)  



def AddNormals1(pd):
    
   

    pdNormals = vtk.vtkPolyDataNormals()
    
    pdNormals.SetInputData(pd)
    pdNormals.ComputePointNormalsOn()
    pdNormals.SetFeatureAngle (23)
    pdNormals.SplittingOn ()
    pdNormals.Update()
    
    
    
    
    normPoints = pdNormals.GetOutput().GetPointData().GetNormals()
    #areaCells=pdNormals.GetOutput().GetCellData().GetArray(0)
    
    numPoints =  pdNormals.GetOutput().GetNumberOfPoints()
    #data = pdNormals.GetOutput()
    
    print "\n", "number of cells", numPoints
    # convert normal vectors to 3 scalar arrays
    
    data1Pt = vtk.vtkFloatArray()
    data1Pt.SetNumberOfComponents(1)
    data1Pt.SetName("xn")
    
    data2Pt = vtk.vtkFloatArray()
    data2Pt.SetNumberOfComponents(1)
    data2Pt.SetName("yn")
    
    data3Pt = vtk.vtkFloatArray()
    data3Pt.SetNumberOfComponents(1)
    data3Pt.SetName("zn")
    
    
 
    for i in range (0,numPoints):
    
        
        ##print "\narea", area,"\n"
        n0=normPoints.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"
    
        data1Pt.InsertValue(i,x)
        data2Pt.InsertValue(i,y)
        data3Pt.InsertValue(i,z)
    
        
    
    
    pdNormals.GetOutput().GetPointData().AddArray(data1Pt)
    pdNormals.GetOutput().GetPointData().AddArray(data2Pt)
    pdNormals.GetOutput().GetPointData().AddArray(data3Pt)
    
    
    
    
    
    
    
    return(pdNormals.GetOutput())  


def LogPLY(pd,filename):

    writer_log = vtk.vtkPLYWriter()
    writer_log.SetInputData(pd)
    writer_log.SetFileTypeToASCII ()
  
  
    writer_log.SetFileName(filename)
    writer_log.Write()
 


def BandedContour(pd,bcv1,bcv2,bcv3):

    dirName="C:\\temp\\"    
#    cntr=vtk.vtkXMLPolyDataReader()
#    
#    fileName="gr_new_curvmax.vtp"
#    fileName="hankook_new_curvmax.vtp"
#    fileName="hankook_new_DH01_curvmax.vtp"
#    fileName="goodrich_worn_curvmax.vtp"
#    #fileName="goodrich_new_curvmax.vtp"
#    #fileName="ecopia_new_curvmax.vtp"
#    
#    
#    
#    
#    cntr.SetFileName(dirName+fileName)
#    cntr.Update()
    pd.GetPointData().SetActiveScalars("curvature")
    print "\n npts", pd.GetNumberOfPoints()
    bcfltr=vtk.vtkBandedPolyDataContourFilter()
    bcfltr.SetInputData(pd)
    #bcfltr.SetValue(1,300)
    #bcfltr.SetValue(2,100)
    bcfltr.GenerateContourEdgesOn()
    bcfltr.GenerateValues(bcv1,bcv2,bcv3)
    bcfltr.Update()
    LogVTK(bcfltr.GetOutput(), dirName+"bcnfltr.vtp")
    
    return(bcfltr.GetOutput())


 

def SetDirectoryForLog(dirNameForLog):
    
    global directoryForLog
    directoryForLog = dirNameForLog

    if not os.path.exists(directoryForLog):
        os.makedirs(directoryForLog)
        
    print "\n&&&&&&&&&&&&&&&&&&", directoryForLog


def TestMe1():
    
    dir_worn="C:\\Temp\\m1.ply_1632450_\\"
    dir_new="C:\\Temp\\m1.ply_65475_\\"
    
    fnamew="allmesh_clipped_to_grooves.vtp"
 
    #fname="After_clipped.vtp"
    pname="pts_registration.vtp"
    
    worn_mesh_fname=dir_worn+fname
    new_mesh_fname=dir_new+fname
    
    worn_mesh_pname=dir_worn+pname
    new_mesh_pname=dir_worn+pname
    
    
    worn_mesh_tire= vtk.vtkXMLPolyDataReader()
    worn_mesh_tire.SetFileName(worn_mesh_fname)
    worn_mesh_tire.Update()
    
    new_mesh_tire= vtk.vtkXMLPolyDataReader()
    new_mesh_tire.SetFileName(new_mesh_fname)
    new_mesh_tire.Update()
    
    worn_mesh_pts= vtk.vtkXMLPolyDataReader()
    worn_mesh_pts.SetFileName(worn_mesh_pname)
    worn_mesh_pts.Update()
    
    new_mesh_pts= vtk.vtkXMLPolyDataReader()
    new_mesh_pts.SetFileName(new_mesh_pname)
    new_mesh_pts.Update()
    
    sourcePoints=worn_mesh_pts.GetOutput().GetPoints()
    targetPoints=new_mesh_pts.GetOutput().GetPoints()
        
    
    landmarkTransform=vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(sourcePoints)
    landmarkTransform.SetTargetLandmarks(targetPoints)
    landmarkTransform.SetModeToRigidBody()
    landmarkTransform.Update()
    
    print "\n got here"
    
    icp = vtk.vtkIterativeClosestPointTransform()
    
    icp.SetSource(worn_mesh_tire.GetOutput())
    icp.SetTarget(new_mesh_tire.GetOutput())
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToRMS();
    #icp.DebugOn()
    #icp.SetMaximumMeanDistance      ( 0.002 )
    icp.SetMaximumNumberOfIterations(200)
    
    #icp.SetMaximumNumberOfIterations(4)
    #icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    
    print "\n got here b"
    print icp.GetMeanDistance()
    
    print icp.GetLandmarkTransform()
    
    print icp.GetLandmarkTransform().GetMatrix()
    
 
    
    
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(worn_mesh_tire.GetOutput())
    
    icpTransformFilter.SetTransform(icp)
    #icpTransformFilter.SetTransform(icp)
    
    icpTransformFilter.Update()
    
    #transformedSource = icpTransformFilter.GetOutput()
    
    LogVTK(icpTransformFilter.GetOutput(), "c:\\temp\\wornmeshreg1.vtp")
    #    writer_td60 = vtk.vtkXMLPolyDataWriter()
    #    writer_td60.SetInputData(icpTransformFilter.GetOutput())
    #    writer_td60.SetFileName(directoryForLog+"rotated_mesh_after_pcaandicp.vtp")
    #    writer_td60.Write()
      
    #dontcare,avgdist=ComputeDistance(7777,meshImprove,TransformPD(esmeshImproveRotate180.GetOutput(), icp),0 , True )    
    
    #print "\n $$$$$$$$$$$$$$$$$$$$$$$$ avg dist is ", avgdist 
def TestMe(dir_worn,dir_new,fnamev,fnamew, sourcePoints,targetPoints):
    
    #dir_worn="C:\\Temp\\m1.ply_1632450_\\"
    #dir_new="C:\\Temp\\m1.ply_65475_\\"
    
    #fnamew="allmesh_clipped_to_grooves.vtp"
 
    #fname="After_clipped.vtp"
    pname="pts_registration.vtp"
    
    worn_mesh_fname_all=dir_worn+fnamev
    new_mesh_fname_all=dir_new+fnamev
    
    worn_mesh_fname=dir_worn+fnamew
    new_mesh_fname=dir_new+fnamew
#    
#    worn_mesh_pname=dir_worn+pname
#    new_mesh_pname=dir_worn+pname
#    
    
    worn_mesh_tire_all= vtk.vtkXMLPolyDataReader()
    worn_mesh_tire_all.SetFileName(worn_mesh_fname_all)
    worn_mesh_tire_all.Update()
    
    worn_mesh_tire= vtk.vtkXMLPolyDataReader()
    worn_mesh_tire.SetFileName(worn_mesh_fname)
    worn_mesh_tire.Update()
    
    new_mesh_tire_all= vtk.vtkXMLPolyDataReader()
    new_mesh_tire_all.SetFileName(new_mesh_fname_all)
    new_mesh_tire_all.Update()
    
    new_mesh_tire= vtk.vtkXMLPolyDataReader()
    new_mesh_tire.SetFileName(new_mesh_fname)
    new_mesh_tire.Update()
    print "|n 55555555555555555555555555 new_mesh_tire",new_mesh_tire.GetOutput().GetNumberOfPoints()
#    
#    worn_mesh_pts= vtk.vtkXMLPolyDataReader()
#    worn_mesh_pts.SetFileName(worn_mesh_pname)
#    worn_mesh_pts.Update()
#    
#    new_mesh_pts= vtk.vtkXMLPolyDataReader()
#    new_mesh_pts.SetFileName(new_mesh_pname)
#    new_mesh_pts.Update()
#    
#    sourcePoints=worn_mesh_pts.GetOutput().GetPoints()
#    targetPoints=new_mesh_pts.GetOutput().GetPoints()
        
    
    landmarkTransform=vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(sourcePoints)
    landmarkTransform.SetTargetLandmarks(targetPoints)
    landmarkTransform.SetModeToRigidBody()
    landmarkTransform.Update()
    
    worn_mesh_tire_lmtr=TransformPD(worn_mesh_tire.GetOutput(), landmarkTransform)
    LogVTK(worn_mesh_tire_lmtr.GetOutput(), "c:\\temp\\wornmeshtire.vtp")
    
    
    print "\n got here"
    #LogVTK(sourcePoints.GetOutput(), "c:\\temp\\sourcepts.vtp")
    #LogVTK(targetPoints.GetOutput(), "c:\\temp\\targetpts.vtp")
    #LogVTK(worn_mesh_tire.GetOutput(), "c:\\temp\\worn_lmt.vtp")
    
    icp = vtk.vtkIterativeClosestPointTransform()
    
    icp.SetSource(worn_mesh_tire_lmtr.GetOutput())
    icp.SetTarget(new_mesh_tire.GetOutput())
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToRMS();
    #icp.DebugOn()
    #icp.SetMaximumMeanDistance      ( 0.002 )
    icp.SetMaximumNumberOfIterations(200)
    
    #icp.SetMaximumNumberOfIterations(4)
    #icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    
    print "\n got here b"
    print icp.GetMeanDistance()
    
    print icp.GetLandmarkTransform()
    
    print icp.GetLandmarkTransform().GetMatrix()
    
 
    
    
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(worn_mesh_tire_lmtr.GetOutput())
    
    icpTransformFilter.SetTransform(icp)
    #icpTransformFilter.SetTransform(icp)
    
    icpTransformFilter.Update()
    
    #transformedSource = icpTransformFilter.GetOutput()
    
    LogVTK(icpTransformFilter.GetOutput(), "c:\\temp\\wornmeshreg.vtp")
    pcaTransform=FileToTransform(dir_new+"orientationToXYZTransform.txt")
    halfTransformBack=FileToTransform(dir_new+"halfTransformBack.txt")
    LogVTK(TransformPD(worn_mesh_tire_lmtr.GetOutput(), icp).GetOutput(), "c:\\temp\\wornmeshreg1.vtp")
    LogVTK(TransformPD(icpTransformFilter.GetOutput(), pcaTransform).GetOutput(), "c:\\temp\\wornmeshregpca.vtp")
    LogVTK(TransformPD(TransformPD(icpTransformFilter.GetOutput(), pcaTransform).GetOutput(),halfTransformBack).GetOutput() , "c:\\temp\\wornmeshregpcahalfback.vtp")
    LogVTK(TransformPD(TransformPD(worn_mesh_tire_all.GetOutput(), landmarkTransform).GetOutput(),icp).GetOutput() , "c:\\temp\\wornmeshallreg.vtp")
    LogVTK(TransformPD(TransformPD(TransformPD(worn_mesh_tire_all.GetOutput(), landmarkTransform).GetOutput(),icp).GetOutput(),pcaTransform).GetOutput() , "c:\\temp\\wornmeshallpca.vtp")
    LogVTK(TransformPD(TransformPD(TransformPD(TransformPD(worn_mesh_tire_all.GetOutput(), landmarkTransform).GetOutput(),icp).GetOutput(),pcaTransform).GetOutput(),halfTransformBack).GetOutput() , "c:\\temp\\wornmeshallpcaback.vtp")
    
    
    LogVTK(TransformPD(TransformPD(icpTransformFilter.GetOutput(), pcaTransform).GetOutput(),halfTransformBack).GetOutput() , "c:\\temp\\wornmeshregpcahalfback.vtp")
    LogVTK(TransformPD(TransformPD(new_mesh_tire_all.GetOutput(), pcaTransform).GetOutput(),halfTransformBack).GetOutput() , "c:\\temp\\newmeshregpcahalfback.vtp")
    
    #    writer_td60 = vtk.vtkXMLPolyDataWriter()
    #    writer_td60.SetInputData(icpTransformFilter.GetOutput())
    #    writer_td60.SetFileName(directoryForLog+"rotated_mesh_after_pcaandicp.vtp")
    #    writer_td60.Write()
      
    #dontcare,avgdist=ComputeDistance(7777,meshImprove,TransformPD(esmeshImproveRotate180.GetOutput(), icp),0 , True )    
    
    #print "\n $$$$$$$$$$$$$$$$$$$$$$$$ avg dist is ", avgdist 

def FileToTransform(fileName):
 
    m4x4=np.zeros(shape=(4,4))
    m4x4=np.loadtxt(fileName)
    flat4x4=np.hstack(m4x4)
#    for i in range(4):
#        row1=np.m4x4(i)
#    for 
#    f = open(fileName)
#    lines = f.readlines()
#    #lines = (line.rstrip('\n') for line in open(filename))
#    f.close()
#    transfArray=[]
#    for i,line in enumerate(lines):
#        transformationArray=line.split()
#        transformationArray1=[float(a) for a in transformationArray]
#        print transformationArray
#        transfArray.extend(transformationArray1)
#        print "\n****************\n"
#        print transformationArray1
#    
#        print transfArray
#    
#        matrix4x4=tuple(transfArray)
#    
#        print "\n&&&&&&&&&&&&&&&&&\n",matrix4x4
#    
    
    m= vtk.vtkMatrix4x4()
    m.DeepCopy(flat4x4)
    print "\n M is ",m

    transform =vtk.vtkTransform()
    transform.SetMatrix(m)
    
    return(transform)

def TransformToFile(transform, fileName):
#    print "\n # of transforms",    transform.GetNumberOfConcatenatedTransforms()
#    
#    for i in range(transform.GetNumberOfConcatenatedTransforms()):
#        print transform.GetConcatenatedTransform(i)
    
    #m= vtk.vtkMatrix4x4()
    m1=transform.GetMatrix()
    m4x4=np.zeros(shape=(4,4))
    print "\n m1 is ",m1
    out= [ 0 for i in range(16)]
    #m1.DeepCopy(out)
    print m1,out
    for i in range(4):
        row1=[]
        for j in range(4):
            elem=m1.GetElement(i,j)
            row1.append(elem)
            
            print "\n elem is ",elem
        m4x4[i]=row1
    #m.DeepCopy(matrix4x4)
    print m4x4
    np.savetxt(fileName,m4x4)
       


def FindTrueRadius(radiusInMetersEstimate, xCenter,yCenter, mesh):

# threshold to high curvature
    print "\n****************** FTR ***************\n"
    meshwithnormals=AddNormals(mesh.GetOutput())
    print "\n&&&&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%&&&&\n" ,meshwithnormals.GetOutput().GetNumberOfPoints()
    LogVTK(meshwithnormals.GetOutput(), directoryForLog+"meshwithnormals.vtp")
    
    #meshwithnormalsThresh=ThresholdTread1(meshwithnormals.GetOutput(),"yn", 0.2,1)
    #LogVTK(meshwithnormalsThresh.GetOutput(), directoryForLog+"meshwithnormalsThresh.vtp")

    # threshold to high curvature
    sipesAndGroovesInMesh = vtk.vtkCurvatures()
    sipesAndGroovesInMesh.SetCurvatureTypeToMaximum()
    #sipesAndGroovesInMesh.SetInputConnection(meshwithnormals.GetOutputPort())
    sipesAndGroovesInMesh.SetInputConnection(meshwithnormals.GetOutputPort())
    sipesAndGroovesInMesh.Update()   
    
    sipesAndGroovesInMeshThresh1=ThresholdTread1(sipesAndGroovesInMesh,"yn", 0.1,1.0)
    sipesAndGroovesInMeshThresh=ThresholdTread1(sipesAndGroovesInMeshThresh1,"Maximum_Curvature", 400,10000)
    LogVTK(sipesAndGroovesInMeshThresh.GetOutput(), directoryForLog+"sipesAndGroovesInMeshThresh.vtp")

    phaseOfTire=4
  
   
    startRadius = radiusInMetersEstimate - 0.05
    endRadius = radiusInMetersEstimate + 0.05
#    startRadius = .307
#    endRadius=.607
    print "\n startRadius endRadius", startRadius,endRadius
    

    bbox=[0,0,0,0,0,0]
    bbox=mesh.GetOutput().GetBounds()

    lengthOfSwath=bbox[3]-bbox[2]
#    circumferenceOfCircle=2*math.pi*lengthOfSwath
#    angleSubtendedBySwath= lengthOfSwath/circumferenceOfCircle*360
#    lengthOfOverlap = phaseOfTire/angleSubtendedBySwath*lengthOfSwath
#    
#    print "\n ***********************&&&&&&&&&&&&&&&&&&###################  ", lengthOfSwath,circumferenceOfCircle,
    
#    bbox1=[lengthOfOverlap/2 , lengthOfOverlap/2,bbox[2],bbox[3],bbox[4],bbox[5]]
    
#    clipped,box=ClipRetBB(mesh.GetOutput(),bbox1,None)
#    LogVTK(clipped.GetOutput(), directoryForLog+"clipped1.vtp")

 
    
    minAvgDist=99999
    for radiusTestinMM in range(int(startRadius*1000),int(endRadius*1000),10):
            radiusTestinMeters=float(radiusTestinMM)/1000
            if (yCenter>0):
                transformMeshForMatch=vtk.vtkTransform()
                transformMeshForMatch.PostMultiply()
              
                transformMeshForMatch.RotateZ(-90)
                transformMeshForMatch.RotateX(-90)
                transformMeshForMatch.Translate(0,-radiusTestinMeters,0)
            else:
                transformMeshForMatch=vtk.vtkTransform()
                transformMeshForMatch.PostMultiply()
           
                transformMeshForMatch.RotateY(90)
                transformMeshForMatch.RotateZ(90)
                transformMeshForMatch.Translate(0,radiusTestinMeters,0)
                
            circumferenceOfCircle=2*math.pi*radiusTestinMM/1000
            angleSubtendedBySwath= lengthOfSwath/circumferenceOfCircle*360
            lengthOfOverlap = phaseOfTire/angleSubtendedBySwath*lengthOfSwath
            correctionRotation=angleSubtendedBySwath/2-phaseOfTire/2
            
            print "\n ***********************&&&&&&&&&&&&&&&&&&###################  ", lengthOfSwath,circumferenceOfCircle,angleSubtendedBySwath,lengthOfOverlap,correctionRotation
            
            bbox1=[-lengthOfOverlap/2 , lengthOfOverlap/2,bbox[2],bbox[3],bbox[4],bbox[5]]
            # transform box accoding to clipping amoung


            
            clipped,box=ClipRetBB(mesh.GetOutput(),bbox1,None)
            LogVTK(clipped.GetOutput(), directoryForLog+"clipped1.vtp")

            
        
            topBottomOfCircle=TransformPD(sipesAndGroovesInMeshThresh.GetOutput(),transformMeshForMatch)
            topBottomOfCircleOriginal=TransformPD(mesh.GetOutput(),transformMeshForMatch)
            #print "\n99999999999999999999 of pts", topBottomOfCircleOriginal.GetOutput().GetNumberOfPoints()
            LogVTK(topBottomOfCircle.GetOutput(), directoryForLog+"topBottomOfCircle_"+str(radiusTestinMeters)+".vtp")
            LogVTK(topBottomOfCircleOriginal.GetOutput(), directoryForLog+"topBottomOfCircleOriginal_"+str(radiusTestinMeters)+".vtp")
            bounds = topBottomOfCircle.GetOutput().GetBounds()
            #boundsPlus40=(bounds[1]-sizeOf4DegreeSwath,bounds[1],bounds[2],bounds[3],bounds[4],bounds[5])
            
            # clip the right 
            
            
           
            
            transformMeshForMatchPlus4Degrees=vtk.vtkTransform()
            transformMeshForMatchPlus4Degrees.RotateZ(phaseOfTire)
            topBottomOfCirclePlus4Degrees=TransformPD(topBottomOfCircle.GetOutput(),transformMeshForMatchPlus4Degrees)
            LogVTK(topBottomOfCirclePlus4Degrees.GetOutput(), directoryForLog+"topBottomOfCirclePlus4Degrees"+str(radiusTestinMeters)+".vtp")
            boundsP4D = topBottomOfCirclePlus4Degrees.GetOutput().GetBounds()
            #boundsPlus40=(bounds[1]-sizeOf4DegreeSwath,bounds[1],bounds[2],bounds[3],bounds[4],bounds[5])
            boundsPlus40=(boundsP4D[0],bounds[1],boundsP4D[2],boundsP4D[3],boundsP4D[4],boundsP4D[5])
            #print "\nboundsPlus40", boundsPlusDirection
            # clip the right 
            topBottomOfCirclePlus4DegreesClipped=Clip(topBottomOfCirclePlus4Degrees.GetOutput(), boundsPlus40)
            LogVTK(topBottomOfCirclePlus4DegreesClipped.GetOutput(), directoryForLog+"topBottomOfCirclePlus4DegreesClipped"+str(radiusTestinMeters)+".vtp")
            
            boundsPlusDirection=(boundsP4D[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5])
            print "\nboundsPlusDirection", boundsPlusDirection
            topBottomOfCirclePlusDirectionClipped=Clip(topBottomOfCircle.GetOutput(), boundsPlusDirection)
            LogVTK(topBottomOfCirclePlusDirectionClipped.GetOutput(), directoryForLog+"topBottomOfCirclePlusDirectionClipped"+str(radiusTestinMeters)+".vtp")
        
            transformMeshForMatchPlusCorrectionRotation=vtk.vtkTransform()
            transformMeshForMatchPlusCorrectionRotation.RotateZ(correctionRotation)
            box2=vtk.vtkBox()
            box2.SetBounds(bbox)
            #print "\m # og pts", box2.GetNumberOfPoints()
            rotatedClippingBoxPos=TransformIF(box2,transformMeshForMatchPlusCorrectionRotation)
           
            clippedSwathPos,box=ClipRetBB(topBottomOfCircle.GetOutput(),rotatedClippingBoxPos,None)
            LogVTK(clippedSwathPos.GetOutput(), directoryForLog+"topBottomOfCircleMinusDirectionClipped"+str(radiusTestinMeters)+".vtp")
            
        
        
        
        
            transformMeshForMatchMinus4Degrees=vtk.vtkTransform()
            transformMeshForMatchMinus4Degrees.RotateZ(-phaseOfTire)
            topBottomOfCircleMinus4Degrees=TransformPD(topBottomOfCircle.GetOutput(),transformMeshForMatchMinus4Degrees)
            LogVTK(topBottomOfCircleMinus4Degrees.GetOutput(), directoryForLog+"topBottomOfCircleMinus4Degrees"+str(radiusTestinMeters)+".vtp")
            boundsM4D = topBottomOfCircleMinus4Degrees.GetOutput().GetBounds()
            #boundsMinus40=(bounds[0],bounds[0]+sizeOf4DegreeSwath,bounds[2],bounds[3],bounds[4],bounds[5])
            boundsMinus40=(bounds[0],boundsM4D[1],boundsM4D[2],boundsM4D[3],boundsM4D[4],boundsM4D[5])
            topBottomOfCircleMinus4DegreesClipped=Clip(topBottomOfCircleMinus4Degrees.GetOutput(), boundsMinus40)
            boundsMinusDirection=(bounds[0],boundsM4D[1],bounds[2],bounds[3],bounds[4],bounds[5])
            LogVTK(topBottomOfCircleMinus4DegreesClipped.GetOutput(), directoryForLog+"topBottomOfCircleMinus4DegreesClipped"+str(radiusTestinMeters)+".vtp")
            topBottomOfCircleMinusDirectionClipped=Clip(topBottomOfCircle.GetOutput(), boundsMinusDirection)
            LogVTK(topBottomOfCircleMinusDirectionClipped.GetOutput(), directoryForLog+"topBottomOfCircleMinusDirectionClipped"+str(radiusTestinMeters)+".vtp")
            
    
    
    
    
    
    
    
    
            print "\nCalling Compute Distance from Plus side"
            #differenceMesh,avgDist=ComputeDistance(int(radiusTestinMM),topBottomOfCirclePlus4DegreesClipped,topBottomOfCircleMinus4DegreesClipped,0,True)
            differenceMeshPlus,avgDistPlus=ComputeDistance(int(radiusTestinMM),topBottomOfCirclePlus4DegreesClipped,topBottomOfCirclePlusDirectionClipped,0,True)
            
            print "\nCalling Compute Distance from Minus side"
             #differenceMesh,avgDist=ComputeDistance(int(radiusTestinMM),topBottomOfCirclePlus4DegreesClipped,topBottomOfCircleMinus4DegreesClipped,0,True)
            differenceMeshMinus,avgDistMinus=ComputeDistance(int(radiusTestinMM),topBottomOfCircleMinus4DegreesClipped,topBottomOfCircleMinusDirectionClipped,0,True)
                        
            avgDist=(avgDistPlus+avgDistMinus)/2
            
        
       
       
            if (minAvgDist>avgDist):
                minAvgDist=avgDist
                bestFitRadius=radiusTestinMeters
            print "\n **********  bestFitRadius, minAvgDist, avgDist, avgDistPlus, avgDistPlus ", bestFitRadius, minAvgDist, avgDist,avgDistPlus, avgDistMinus
    print "\n&&&&&&&&&&&&&& rfinal minAvgDist bestFitRadius", minAvgDist,bestFitRadius
    return(bestFitRadius)

def FindTrueRadius1(radiusInMetersEstimate, xCenter,yCenter, mesh):


 
    bbox=[0,0,0,0,0,0]
    bbox=mesh.GetOutput().GetBounds()

    lengthOfSwath=bbox[3]-bbox[2]
        
    normalizeMeshForRadiusCalculation=vtk.vtkTransform()
    #normalizeMeshForRadiusCalculation.Translate(0,0,-bbox[5])
    # hack radius correction
    normalizeMeshForRadiusCalculation.Translate(0,0,-0.012)
    
    mesh=TransformPD(mesh.GetOutput(),normalizeMeshForRadiusCalculation)
    LogVTK(mesh.GetOutput(), directoryForLog+"normalizeMeshForRadiusCalculation"+".vtp")

            

    #normalize top of mesh to 0


# threshold to high curvature
    print "\n****************** FTR ***************\n"
    meshwithnormals=AddNormals(mesh.GetOutput())
    print "\n&&&&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%&&&&\n" ,meshwithnormals.GetOutput().GetNumberOfPoints()
    LogVTK(meshwithnormals.GetOutput(), directoryForLog+"meshwithnormals.vtp")
    
    #meshwithnormalsThresh=ThresholdTread1(meshwithnormals.GetOutput(),"yn", 0.2,1)
    #LogVTK(meshwithnormalsThresh.GetOutput(), directoryForLog+"meshwithnormalsThresh.vtp")

    # threshold to high curvature
    sipesAndGroovesInMesh = vtk.vtkCurvatures()
    sipesAndGroovesInMesh.SetCurvatureTypeToMaximum()
    #sipesAndGroovesInMesh.SetInputConnection(meshwithnormals.GetOutputPort())
    sipesAndGroovesInMesh.SetInputConnection(meshwithnormals.GetOutputPort())
    sipesAndGroovesInMesh.Update()   
    
    sipesAndGroovesInMeshThresh1=ThresholdTread1(sipesAndGroovesInMesh,"yn", 0.1,1.0)
    sipesAndGroovesInMeshThresh=ThresholdTread1(sipesAndGroovesInMeshThresh1,"Maximum_Curvature", 400,10000)
    LogVTK(sipesAndGroovesInMeshThresh.GetOutput(), directoryForLog+"sipesAndGroovesInMeshThresh.vtp")

    phaseOfTire=4.0
  
   
    startRadius = radiusInMetersEstimate - 0.05
    endRadius = radiusInMetersEstimate + 0.05
#    startRadius = .307
#    endRadius=.607
    print "\n startRadius endRadius", startRadius,endRadius
    
#
#    bbox=[0,0,0,0,0,0]
#    bbox=mesh.GetOutput().GetBounds()
#
#    lengthOfSwath=bbox[3]-bbox[2]

    meshInCanonicalForm=sipesAndGroovesInMeshThresh

    
 
    
    minAvgDist=99999
    for radiusTestinMM in range(int(startRadius*1000),int(endRadius*1000),1):
        radiusTestinMeters=float(radiusTestinMM)/1000
         
        transformMeshForMatch=vtk.vtkTransform()
        transformMeshForMatch.Translate(0,0,radiusTestinMeters)
        
            
            
        meshTranslatedToRadius=TransformPD(meshInCanonicalForm.GetOutput(),transformMeshForMatch)
        LogVTK(meshTranslatedToRadius.GetOutput(), directoryForLog+"meshTranslatedToRadius"+str(radiusTestinMeters)+".vtp")
        bbox=[0,0,0,0,0,0]
        bbox=meshTranslatedToRadius.GetOutput().GetBounds()
        
        
        circumferenceOfCircle=2*math.pi*radiusTestinMM/1000
        angleSubtendedBySwath= lengthOfSwath/circumferenceOfCircle*360
        lengthOfOverlap = phaseOfTire/360.0*circumferenceOfCircle
        correctionRotation=angleSubtendedBySwath/2-phaseOfTire/2
        print "\n ***********************&&&&&&&&&&&&&&&&&&###################  ", lengthOfSwath,circumferenceOfCircle,angleSubtendedBySwath,lengthOfOverlap,correctionRotation
          
         
        bbox1=[bbox[0],bbox[1],-lengthOfOverlap/2 , lengthOfOverlap/2,bbox[4],bbox[5]]
            # transform box accoding to clipping amoung

        transformMeshClockwiseToCenterOverlapPortion=vtk.vtkTransform()
        transformMeshClockwiseToCenterOverlapPortion.PostMultiply()
        transformMeshClockwiseToCenterOverlapPortion.RotateX(phaseOfTire/2)
        
        transformMeshCounterClockwiseToCenterOverlapPortion=vtk.vtkTransform()
        transformMeshCounterClockwiseToCenterOverlapPortion.PostMultiply()
        transformMeshCounterClockwiseToCenterOverlapPortion.RotateX(-phaseOfTire/2)
        
        
        meshClockwiseToCenterOverlapPortion=TransformPD(meshTranslatedToRadius.GetOutput(),transformMeshClockwiseToCenterOverlapPortion)
        LogVTK(meshClockwiseToCenterOverlapPortion.GetOutput(), directoryForLog+"meshClockwiseToCenterOverlapPortion"+str(radiusTestinMeters)+".vtp")

 
        meshCounterClockwiseToCenterOverlapPortion=TransformPD(meshTranslatedToRadius.GetOutput(),transformMeshCounterClockwiseToCenterOverlapPortion)
        LogVTK(meshCounterClockwiseToCenterOverlapPortion.GetOutput(), directoryForLog+"meshCounterClockwiseToCenterOverlapPortion"+str(radiusTestinMeters)+".vtp")


       
        meshClockwiseToCenterOverlapPortionClipped=Clip(meshClockwiseToCenterOverlapPortion.GetOutput(), bbox1)
        LogVTK(meshClockwiseToCenterOverlapPortionClipped.GetOutput(), directoryForLog+"meshClockwiseToCenterOverlapPortionClipped"+str(radiusTestinMeters)+".vtp")


        meshCounterClockwiseToCenterOverlapPortionClipped=Clip(meshCounterClockwiseToCenterOverlapPortion.GetOutput(), bbox1)
        LogVTK(meshCounterClockwiseToCenterOverlapPortionClipped.GetOutput(), directoryForLog+"meshCounterClockwiseToCenterOverlapPortionClipped"+str(radiusTestinMeters)+".vtp")
        
        print "\nCalling Compute Distance "
        
        differenceMeshP,avgDist=ComputeDistance(int(radiusTestinMM),meshClockwiseToCenterOverlapPortionClipped,meshCounterClockwiseToCenterOverlapPortionClipped,0,True)
            
       
        if (minAvgDist>avgDist):
            minAvgDist=avgDist
            bestFitRadius=radiusTestinMeters
        print "\n **********  bestFitRadius, minAvgDist, avgDist,", bestFitRadius, minAvgDist, avgDist
   
     
      
            
    return(bestFitRadius)


    
    


def FindZRotationOrientation(radiusInMM,mesh):
    
#    sectionWidth=11.1*25.4
#    aspectRatio=0
#    overallDiameter=41.1*25.4
#    radiusInMM=radiusGuess*25.4
#    
    planeA=vtk.vtkPlane()
    # change Z baxck to 0
    
    planeA.SetOrigin(0,0,0)
    planeA.SetNormal(1,0,0)
    circFrag=vtk.vtkCutter()
    circFrag.SetSortBy(0)
    circFrag.SetCutFunction(planeA)
    circFrag.SetInputData(mesh.GetOutput())
    circFrag.Update()
           
    
    numPts = circFrag.GetOutput().GetNumberOfPoints()
    
    pointsArray = np.zeros(( numPts,2))
    
    print "\n number of Pts", numPts
    
    #circPts = circFrag.GetOutput().GetPointData().GetArray(1)
    
    # pick out Y and Z
    grooveList=[]
    for i in range(numPts):
        #print "i", i
        pt=circFrag.GetOutput().GetPoint(i)
        #print pt[1],pt[2]
        grooveList.append((pt[1],pt[2]) )
        
    #print "***",grooveList    
    #
    for i,item in enumerate(grooveList):
        #print "\n item", item, "\n"
        pointsArray[i,0] = item[0]
        pointsArray[i,1] = item[1]
    
    
    #print "\nPoints Arrayt", pointsArray
    
    model = CircleModel()
    r = radiusInMM/1000
    model.params = (0, r/2, r)
       
    
    if  (pointsArray.size > 0):
        model.estimate(pointsArray)
        
        
        #x = np.arange(-200, 200)
        #y = 0.2 * x + 20
        #data = np.column_stack([x, y])
        model_robust, inliers = ransac(pointsArray, CircleModel, min_samples=10,
                    residual_threshold=0.00001, max_trials=1000)                               
        print "\ninliers",inliers
        print "\circle model params",   model_robust.params,"\n"    
    
    print "\n &&&& Residuals"
    print model.residuals
    #print model.residuals(np.array([[0, 2]]))
    print model.residuals(pointsArray)
    
    print model.residuals(pointsArray)[1]
    
    resid=model.residuals(pointsArray)
    absresid=np.abs(resid)
    print "mean", np.mean(absresid)
    print "max", np.max(absresid), np.median(absresid)
    
    #circle params are Xcenter, Ycenter, Radius
    
    print "\n model robust parasms", model_robust.params
    
    return( (model_robust.params[0], model_robust.params[1] ,model_robust.params[2]),( np.mean(absresid), np.max(absresid), np.median(absresid) ))   

from skimage.measure import LineModel, CircleModel, ransac

def ExtractTireNameComponents(fp):


    
    print os.path.split(fp), "\n"
    # ('/home/aa/bb', 'ff.html')
    
    print os.path.dirname(fp),"\n"
    # /home/aa/bb
    
    print os.path.basename(fp),"\n"
    
    dirPath=os.path.dirname(fp)
    tireFilePathComponents= dirPath.split("\\")
    
    print "tfpc\n", tireFilePathComponents,"\n"
    
    for tireFilePathComponentsItem in tireFilePathComponents:
        if tireFilePathComponentsItem.startswith("Tire_"):
            tireNameComponents=tireFilePathComponentsItem.split("_")
            print "\ntireName", tireNameComponents
            print "\ntireName compoments", tireNameComponents
            mfr=tireNameComponents[1]
            brand=tireNameComponents[2]
            sectionWidth=int(tireNameComponents[3])
            aspectRatio=float(tireNameComponents[4])/100.0
            wheelDiameter=float(tireNameComponents[5])
            newTireFlag=(tireNameComponents[6]=="New")
    
    

    print "\n Section Wdith   Aspct Ratio  WheelDiameter  \n",sectionWidth, aspectRatio, wheelDiameter
    
    return( (mfr, brand, sectionWidth, aspectRatio, wheelDiameter,newTireFlag))

def ThresholdTread(pd,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    print "\n *** Entering Threshold Tread", start, end
    
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname   
        if (aname=="Distance"):
            break
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd)
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    #pdt.SetInputArrayToProcess(0, 0, 0, i, "Distance")
    pdt.SetInputArrayToProcess(i, 0, 0, 0, "Distance")
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    print "\n leaving threshold tread",pd.GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt.GetOutput())
    
def ThresholdTread2(pd,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    print "\n *** Entering Threshold Tread", start, end
    
    numberOfPointArrays = pd.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetOutput().GetPointData().GetArrayName(i)
        print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname    
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd.GetOutput())
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    pdt.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    print "\n leaving threshold tread",pd.GetOutput().GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt)  

def ThresholdTreadScalar(pd,scalarName,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    #print "\n *** Entering Threshold Tread scalar name", start, end,scalarName
    
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    #print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname    
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd)
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    pdt.SetInputArrayToProcess(0, 0, 0, 0, scalarName)
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    #print "\n leaving threshold tread",pd.GetOutput().GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt.GetOutput())

def ThresholdTread1(pd,scalarName,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    #print "\n *** Entering Threshold Tread scalar name", start, end,scalarName
    
    numberOfPointArrays = pd.GetOutput().GetPointData().GetNumberOfArrays()
    #print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetOutput().GetPointData().GetArrayName(i)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname    
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd.GetOutput())
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    pdt.SetInputArrayToProcess(0, 0, 0, 0, scalarName)
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    #print "\n leaving threshold tread",pd.GetOutput().GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt)

def Threshold(pd,scalarName,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    print "\n *** Entering Threshold Tread scalar name", start, end,scalarName
    
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname  
        
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd)
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    pdt.SetInputArrayToProcess(0, 0, 0, 0, scalarName)
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    print "\n leaving threshold tread",pd.GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt.GetOutput())
    
def ThresholdPointOrCellData (pd,bPointData,scalarName,start,end):
    # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    print "\n *** Entering Threshold Tread scalar name", start, end,scalarName

    if bPointData:
        dataToUse = pd.GetPointData()
        fieldAssoc=0
    else:
        dataToUse = pd.GetCellData()
        fieldAssoc=1

    numberOfArrays = dataToUse.GetNumberOfArrays()
    
    print "\n cellorpoint bool, # of arrays  ", bPointData, numberOfArrays,"\n"
    for i in range(numberOfArrays):
        aname = dataToUse.GetArrayName(i)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname   
        if (aname==scalarName):
            break
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd)
    pdt.ThresholdBetween(start,end)
    pdt.SetInputArrayToProcess(0, 0, 0, fieldAssoc, scalarName)
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    print "\n leaving threshold tread",pd.GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt.GetOutput())
    
def ThresholdWithComponent(pd,scalarName,componentNumber,start,end):
       # need to clean up grooves a bit first
    # find max tread depth and offset accordingly by thresholdx
    #print "\n entering threshold tread", 
    print "\n *** Entering Threshold Tread scalar name", start, end,scalarName
    
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname    
    
    
    pdt = vtk.vtkThreshold()
    pdt.SetInputData(pd)
    pdt.SetSelectedComponent	(componentNumber)
    pdt.ThresholdBetween(start,end)
        #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    pdt.SetInputArrayToProcess(0, 0, 0, 0, scalarName)
    pdt.Update()
    espdt = vtk.vtkDataSetSurfaceFilter()
    espdt.SetInputConnection(pdt.GetOutputPort())
    espdt.Update()
    print "\n leaving threshold tread",pd.GetNumberOfPoints(), espdt.GetOutput().GetNumberOfPoints()
    return(espdt.GetOutput())



def FindArrayIndex(pd,scalarName):
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    indexValue=-1
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", i , aname   
        if (aname==scalarName):
            indexValue=i
            break
    
    return(indexValue)
    

def ClipPD(pd,bounds):
    
    box=vtk.vtkBox()
    box.SetBounds(bounds)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(1)
    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper.GetOutput())  


def ClipPD1(pd,bounds,bInsideOut):
    
    box=vtk.vtkBox()
    box.SetBounds(bounds)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(bInsideOut)
    clipper.GenerateClipScalarsOn	()	

    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper.GetOutput())  
    
def ClipPDPlane(pd,pln,bInsideOut):
    
  

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(pln)
    clipper.SetInsideOut(bInsideOut)
    clipper.GenerateClipScalarsOn	()	

    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper.GetOutput())  
    
    


    
def ClipPDBox(pd,box,bInsideOut):
    

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(bInsideOut)
    clipper.GenerateClipScalarsOn	()	

    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper.GetOutput())  


def Clip(pd,bounds):
    
    box=vtk.vtkBox()
    box.SetBounds(bounds)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(1)
    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper)
    


def ClipInsideOut(pd,bounds):
    
    box=vtk.vtkBox()
    box.SetBounds(bounds)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(0)
    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    return(esClipper)
    
    
def ClipRetBB(pd,bounds,box):
    
    if box is None:
        box=vtk.vtkBox()
        box.SetBounds(bounds)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(1)
    clipper.Update()
    
    esClipper = vtk.vtkDataSetSurfaceFilter()
    esClipper.SetInputConnection(clipper.GetOutputPort())
    esClipper.Update()
    
    return(esClipper,box)
    


def TransformPD(pd,transform1):
    transfmesh = vtk.vtkTransformPolyDataFilter()
    #transfmesh.SetInputConnection(reader.GetOutputPort())
    transfmesh.SetInputData(pd)
    
    transfmesh.SetTransform(transform1)
    transfmesh.Update()
    es_transfmesh = vtk.vtkDataSetSurfaceFilter()
    es_transfmesh.SetInputConnection(transfmesh.GetOutputPort())
    es_transfmesh.Update()
    return(es_transfmesh)
    

    
    
def SurfaceToDO(surface):
  es = vtk.vtkDataSetSurfaceFilter()
  es.SetInputConnection(surface.GetOutputPort())
  es.Update()

def LogVTK(pd,filename):

    writer_log = vtk.vtkXMLPolyDataWriter()
    writer_log.SetInputData(pd)
  
    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
    writer_log.SetFileName(filename)
    writer_log.Write()


def SeparateMeshIntoBlocks1(mesh,ner):
    
#    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
#    connectivityFilter.SetInputData(mesh)
#    
#    connectivityFilter.SetExtractionModeToSpecifiedRegions()
#    connectivityFilter.ColorRegionsOn()
#    
#    connectivityFilter.ColorRegionsOn()
#    connectivityFilter.Update()
#    
#
#    # we are assuming at this point that the ONLY components aretread blocks    
#    ner= connectivityFilter.GetNumberOfExtractedRegions ()
#    print "\n number of regions\n ", int(ner)
#    
    blockList=[]
    for i in range(ner):
         #connectivityFilter.AddSpecifiedRegion(i)
         meshRegion=Threshold(mesh,"leftBound", i,i)
         #ta.Threshold(pd4,"xn",-0.5,0.5)
#         
#         cf1 = vtk.vtkCleanPolyData()
#         cf1.SetInputConnection(mesh)
#         cf1.Update()
         numPts =  meshRegion.GetOutput().GetNumberOfPoints()
         print "\nnum pts in current region ", i, numPts
         LogVTK(meshRegion.GetOutput(), directoryForLog+"meshLimitedToRegion_"+str(i)+".vtp")
         
         
        #copy the polydata information without the polylines
         
         pts = vtk.vtkPoints()
         for j in range (meshRegion.GetOutput().GetNumberOfPoints()):
              pt = meshRegion.GetOutput().GetPoint(j)
              pts.InsertNextPoint(pt[0],pt[1],pt[2])
         cellsTriangles = vtk.vtkCellArray()
         cellsTriangles=meshRegion.GetOutput().GetPolys()
        

         pd = vtk.vtkPolyData()
   
         pd.SetPolys(cellsTriangles)
         pd.SetPoints(pts)
         
         LogVTK(pd, directoryForLog+"regionDebug2WithoutStiches_"+str(i)+".vtp")
         blockList.append(pd)

#         
#         connectivityFilter.DeleteSpecifiedRegion(i)
#         
    return(blockList)
         # 
            
    

def SeparateMeshIntoBlocks(mesh):
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(mesh)
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    

    # we are assuming at this point that the ONLY components aretread blocks    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    blockList=[]
    for i in range(ner):
         connectivityFilter.AddSpecifiedRegion(i)
         cf1 = vtk.vtkCleanPolyData()
         cf1.SetInputConnection(connectivityFilter.GetOutputPort())
         cf1.Update()
         numPts =  cf1.GetOutput().GetNumberOfPoints()
         print "\nnum pts in current region ", i, numPts
         LogVTK(cf1.GetOutput(), directoryForLog+"regionDebugStiched_"+str(i)+".vtp")
         
         
        #copy the polydata information without the polylines
         
         pts = vtk.vtkPoints()
         for j in range (cf1.GetOutput().GetNumberOfPoints()):
              pt = cf1.GetOutput().GetPoint(j)
              pts.InsertNextPoint(pt[0],pt[1],pt[2])
         cellsTriangles = vtk.vtkCellArray()
         cellsTriangles=cf1.GetOutput().GetPolys()
        

         pd = vtk.vtkPolyData()
   
         pd.SetPolys(cellsTriangles)
         pd.SetPoints(pts)
         
         LogVTK(pd, directoryForLog+"regionDebug2WithoutStiches_"+str(i)+".vtp")
         blockList.append(pd)

         
         connectivityFilter.DeleteSpecifiedRegion(i)
         
    return(blockList)
         # 
         
         
         
    
def LocateGrooveBoundaries(pd):
    
   
    print "\n%% Entering LocateGrooveBoundaries"
    print "\nStart of Locate Groove# of cells in meshNoGrooves\n", pd.GetNumberOfCells()
    
    LogVTK(pd, directoryForLog+"meshNoGroovesInBegOfLGB.vtp")
    
    # start and end the slicing using the offset
    
#    
#    for(vtkIdType i = 0; i < numberOfPointArrays; i++)
#    {
#    // The following two lines are equivalent
#    //arrayNames.push_back(polydata->GetPointData()->GetArray(i)->GetName());
#    //arrayNames.push_back(polydata->GetPointData()->GetArrayName(i));
#    int dataTypeID = polydata->GetPointData()->GetArray(i)->GetDataType();
#    std::cout << "Array " << i << ": " << polydata->GetPointData()->GetArrayName(i)
#              << " (type: " << dataTypeID << ")" << std::endl;
#    }
    
    
    pts = vtk.vtkPoints()
    for i in range (pd.GetNumberOfPoints()):
        pt = pd.GetPoint(i)
        pts.InsertNextPoint(pt[0],pt[1],pt[2])
    
    cellsTriangles = vtk.vtkCellArray()
    cellsTriangles=pd.GetPolys()
 
    
#   
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    for i in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(i)
        print "\n Array Name for meshNoGrooves", i , aname
    
    horizontalOffset=0.003
    
    # premise:  you can recognize the start and the end of the groove by either NO points in slice
    # OR by a slice with a few points (in which case you're slicing a treadmarker)
    numPtsInSliceThreshold=200
    
#    #filename="c:\\temp\\regionsaggregated1.vtp"
#    meshNoGrooves=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(filename)
#    reader.Update()
    
    bbox=[0,0,0,0,0,0]
    bbox=pd.GetBounds()
    numcuts=30
    minGrooveWidth=0.003
    densityOfLinesForPolyline=5
    
    minX=bbox[0]+horizontalOffset
    maxX=bbox[1]-horizontalOffset
    xInterval=(maxX-minX)/numcuts
    widthForPolyLine=0.0001
    
 

    onBlock=False
    polyLineList=[]
    for i in range(0,numcuts):
        
  
#        print "\n ### cut # ", i
#        print "\ncut pt x", minX+xInterval*float(i)
#        
      
        #select the hull pts and export them
        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::Points
        selection.SetArrayName("X_coord")
        #selection.AddThreshold(0.10, 0.121)
        
        lowX=minX+xInterval*float(i)
        highX=lowX+widthForPolyLine
        
        #selection.AddThreshold(0.12,0.121)
        selection.AddThreshold(lowX,highX)
        selection.Update()
    
        
        extractSelection =vtk.vtkExtractSelection()
        extractSelection.SetInputData(0,pd )
        extractSelection.SetInputData(1, selection.GetOutput())
        extractSelection.Update()

        esx4 = vtk.vtkDataSetSurfaceFilter()
        esx4.SetInputConnection(extractSelection.GetOutputPort())
        esx4.Update()
        
        numPts = esx4.GetOutput().GetNumberOfPoints()
        print "\n Numpts and onblock ",numPts, onBlock
        #selectionNode.SetFieldType(1)

    
        
        if (numPts>=numPtsInSliceThreshold):

                
                selectVertSwathList=[]
                
                origPtIDArray = esx4.GetOutput().GetPointData().GetArray(3)
                
                # sort selectVertSwathList points by Y
                for j in range (esx4.GetOutput().GetNumberOfPoints()):
                    if (j%densityOfLinesForPolyline==0):
                        pt = esx4.GetOutput().GetPoint(j)
                        origPtID=int(origPtIDArray.GetTuple(j)[0])
                        pt1=pd.GetPoint(origPtID)
                        selectVertSwathList.append((j,origPtID,pt[0],pt[1],pt[2],pt1[0],pt1[1],pt1[2]))
                
                # sort points by Y value
                selectVertSwathList.sort(key=lambda tup: tup[3])
          
                LogVTK(esx4.GetOutput(), directoryForLog+"vertcut"+str(i)+".vtp")
        
                #print "\n got here A\n"
                polyLine = vtk.vtkPolyLine()   
                polyLine.GetPointIds().SetNumberOfIds( len(selectVertSwathList))
                #print "\n selectVertSwathList", len(selectVertSwathList)
                
                for j,selectVertSwathListItem in enumerate(selectVertSwathList):
                    #print "\n ################# cut #  and pointid and selectVertSwathList", i,j,selectVertSwathListItem,"\n"
                    origPtID=selectVertSwathListItem[1]          
                    #print "\n ********** Original Pt is is \n", int(origPtID)
                    polyLine.GetPointIds().SetId(j,int(origPtID))
                    
                polyLineList.append(polyLine)  
                print "\n ********* For cut i  # of pts in selectVertSwatch esx4out and Polylne", len(selectVertSwathList), esx4.GetOutput().GetNumberOfPoints(), polyLine.GetNumberOfPoints(),"**************\n"

    
    cells = vtk.vtkCellArray()
    for i,polyLineListItem in enumerate(polyLineList):
        cells.InsertNextCell(polyLineListItem)
        
  
    pd = vtk.vtkPolyData()
    pd.SetLines(cells)
    pd.SetPolys(cellsTriangles)
    pd.SetPoints(pts)
       
       #    meshNoGrooves.GetOutput().SetLines(cells) 
    
    print "\n#End of Locate Groove Boundaries -  of cells in meshNoGrooves\n", pd.GetNumberOfCells()
    LogVTK(pd, directoryForLog+"polylinesAdded_"+ ".vtp")     
    
    return(pd)
   

 

def CopyPtID(pd):

    print "\n Entering CopyPtID"    
    
    intValue= vtk.vtkIntArray()
   
    intValue.SetNumberOfComponents(1)
    intValue.SetName("OldID");       
    
    for i in range(pd.GetNumberOfPoints()):
           intValue.InsertNextValue(i)
        
    pd.GetPointData().AddArray(intValue);
    
    
    floatValue= vtk.vtkFloatArray()
   
    floatValue.SetNumberOfComponents(1)
    floatValue.SetName("X_coord");      
    
    for i in range(pd.GetNumberOfPoints()):
            pt=pd.GetPoint(i)    
            floatValue.InsertNextValue(pt[0] ) # get y value
  
  
    pd.GetPointData().AddArray(floatValue);
    
    

    LogVTK(pd, directoryForLog+"meshWithOldID"+".vtp")
    return(pd)                   
                    


def GetProfileTreadBlocks(boxList):
    
    print "\n%% Entering GetProfileTreadBlocks"
    

    
    bbox=[0,0,0,0,0,0]
    numcuts=25
        
    blockProfiles=[]
    
    blockProfilesVTK=[]
    
    print "************ length of boxlist ", len(boxList)
    
    for h,boxListItem in enumerate(boxList): 
        
        LogVTK(boxListItem, directoryForLog+"boxlistItem"+"_"+ str(h) +"_.vtp")
        
        sliceList=[]
        bbox=boxListItem.GetBounds()
        minY=bbox[2]
        maxY=bbox[3]
        yInterval=(maxY-minY)/numcuts
        blockProfiles.append([])
      
        
        leftVTKPoints = vtk.vtkPoints()
        rightVTKPoints=vtk.vtkPoints()
 
        leftVTKPolyVertex=vtk.vtkPolyVertex()
        rightVTKPolyVertex=vtk.vtkPolyVertex()
        
       
       
        numVTKPts=0
        for i in range(0,numcuts):
            
            sliceList.append(i)
            sliceList[i]=[]
          
#            print "\n ### cut # ", i
#            print "\ncut pt y", minY+yInterval*float(i)
                    
            planeA=vtk.vtkPlane()
            # change Z baxck to 0
            planeA.SetOrigin(0,minY+yInterval*float(i),0)
            planeA.SetNormal(0,-1,0)
            cutterA=vtk.vtkCutter()
            cutterA.SetSortBy(0)
            cutterA.SetCutFunction(planeA)
            cutterA.SetInputData(boxListItem)
            cutterA.Update()
       
            numPts = cutterA.GetOutput().GetNumberOfPoints()
            if (numPts==0):
                continue
            numVTKPts=numVTKPts+1
            #print "\n numPts", numPts
#            writer_cutter = vtk.vtkXMLPolyDataWriter()
#            writer_cutter.SetInputData(cutterA.GetOutput())
#            fname="c:\\temp\\slice_"+str(h)+ "_"+str(i)+"_"+".vtp"
#            writer_cutter.SetFileName(fname)
#            writer_cutter.Write()
                        
                   
            ptsList=[]
            # add points in cut
            for j in range (0,numPts):
#                print "\n j =", j, "\n"
                pt = cutterA.GetOutput().GetPoint(j)
                ptsList.append((pt[0],pt[1],pt[2]))
                #print ptsList
            # actually just need min and max
            ptsList.sort(key=lambda tup: tup[0])
            leftPt = (ptsList[0][0],ptsList[0][1],ptsList[0][2] )
            rightPt = (ptsList[numPts-1][0],ptsList[numPts-1][1],ptsList[numPts-1][2] )
            blockProfiles[h].append( (leftPt,rightPt))
            
            leftVTKPoints.InsertNextPoint(ptsList[0][0],ptsList[0][1],ptsList[0][2])
            rightVTKPoints.InsertNextPoint(ptsList[numPts-1][0],ptsList[numPts-1][1],ptsList[numPts-1][2] )
            
#        
#        leftVTKPolyline.GetPointIds().SetNumberOfIds(numVTKPts)
#        rightVTKPolyline.GetPointIds().SetNumberOfIds(numVTKPts)
#        for r in range(numVTKPts):
#            leftVTKPolyline.GetPointIds().SetId(r,r)
#            rightVTKPolyline.GetPointIds().SetId(r,r)
        
        cellsLeft=vtk.vtkCellArray()
        #cellsLeft.InsertNextCell(leftVTKPolyline)
        cellsLeft.InsertNextCell(leftVTKPolyVertex)
        cellsRight=vtk.vtkCellArray()
#        cellsRight.InsertNextCell(rightVTKPolyline)
        cellsRight.InsertNextCell(rightVTKPolyVertex)
        
          
        leftVTKPolyData= vtk.vtkPolyData()
        leftVTKPolyData.SetPoints(leftVTKPoints)  
#        leftVTKPolyData.SetLines(cellsLeft)
        
     
        
        LogVTK(leftVTKPolyData, directoryForLog+"blockProfileLeft"+"_"+ str(h) +"_.vtp")
#        writer_td89 = vtk.vtkXMLPolyDataWriter()
#        writer_td89.SetInputData(leftVTKPolyData)
#        filename = directoryForLog+"blockProfileLeft"+"_"+ str(h) +"_.vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td89.SetFileName(filename)
#        writer_td89.Write()
        
               
        rightVTKPolyData= vtk.vtkPolyData()
        rightVTKPolyData.SetPoints(rightVTKPoints)  
#        rightVTKPolyData.SetLines(cellsRight)
#        rightVTKPolyData.SetLines(cellsRight)
        
        
        rightVTKPolyData.Update()
        
        
        blockProfilesVTK.append((leftVTKPolyData,rightVTKPolyData))
        
        LogVTK(rightVTKPolyData, directoryForLog+"blockProfileRight"+"_"+ str(h) +"_.vtp")
            
#        writer_td90 = vtk.vtkXMLPolyDataWriter()
#        writer_td90.SetInputData(rightVTKPolyData)
#        filename = directoryForLog+"blockProfileRight"+"_"+ str(h) +"_.vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td90.SetFileName(filename)
#        writer_td90.Write()
    
#
#    print blockProfiles[0][0],blockProfiles[0][1],blockProfiles[0][2]
#    print blockProfiles[1][0],blockProfiles[1][1],blockProfiles[1][2]
#    print blockProfiles[2][0],blockProfiles[2][1],blockProfiles[2][2]
#    print blockProfiles[3][0],blockProfiles[3][1],blockProfiles[3][2]
#    
#    print "\n lengt of block prof", len(blockProfiles)    
#    
#       
#    
#    for q,blockProfilesItem in enumerate(blockProfiles):
#        for r, lrpt in enumerate(blockProfilesItem):
#            print "\n********************\n", q,r, lrpt
#       
#            print "\n\n"
    print "************ length of boxlist ", len(boxList)
    
    return(blockProfiles)  
#
def ComputeIndividualGrooveDistances(grooveTopSideProfiles,clippedMesh):
    
#    
#    filename="c:\\temp\\blockProfileRight_4_.vtp"
#    ptset1=vtk.vtkXMLPolyDataReader()
#    ptset1.SetFileName(filename)
#    ptset1.Update()
#    
#    filename="c:\\temp\\blockProfileLeft_4_.vtp"
#    ptset2=vtk.vtkXMLPolyDataReader()
#    ptset2.SetFileName(filename)
#    ptset2.Update()
#    
#    pointsForHull = vtk.vtkAppendPolyData()
#    pointsForHull.AddInputConnection(ptset1.GetOutputPort() )
#    pointsForHull.AddInputConnection( ptset2.GetOutputPort() )
#    pointsForHull.Update()
#    
#    print "\n # init test pts item appended", pointsForHull.GetOutput().GetNumberOfPoints()
#    print "\n # ttttttttttd", ptset1.GetNumberOfPoints(),ptset2.GetNumberOfPoints(),
#
#    
    grooveBoundaryAsPDList=[]
    numCuts=len(grooveTopSideProfiles[0])
    
    print "\n **************************** GrooveTopSideProfiles"
    for i,grooveTopSideProfilesItem in enumerate(grooveTopSideProfiles):
        print "\n ****************************", "\n",i,grooveTopSideProfilesItem
        
    print "\n numcuts = ", numCuts
    print "\n ****** lengths"
    # debug startement assumes 4 grooves
    #print len(grooveTopSideProfiles[0]), len(grooveTopSideProfiles[1]),len(grooveTopSideProfiles[2]),len(grooveTopSideProfiles[3]),len(grooveTopSideProfiles[4])
##    
#    print "\n # pts", grooveTopSideProfiles[1][1].GetNumberOfPoints()
    
# change the 25 so that it's no hard coded    
    print "\n ^^^*********************************^ Len len(grooveTopSideProfiles))", len(grooveTopSideProfiles)
    
    
    for q in range(1,len(grooveTopSideProfiles)):
    
            # append the points
        pointsOnBoundaryOfGroove = vtk.vtkPoints()
        for r,grooveBoundaryPt in enumerate(grooveTopSideProfiles[q]):
          
            #pt = leftGrooveBoundaryPt[0]
            #print "\n leftGrooveBoundaryPt", grooveBoundaryPt[0],"\n"
          #changed from from 1 to 0 and below from 0 to 1
            pointsOnBoundaryOfGroove.InsertNextPoint(grooveBoundaryPt[0])
        for r,grooveBoundaryPt in enumerate(grooveTopSideProfiles[q-1]):
            #pt = rightGrooveBoundaryPt[1]
            pointsOnBoundaryOfGroove.InsertNextPoint(grooveBoundaryPt[1])    
        
    

        triangleList=[]
        
        for r in range(24):
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0,r)
            tri.GetPointIds().SetId(1,r+numCuts)
            tri.GetPointIds().SetId(2,r+1)
            triangleList.append(tri)
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0,r+1)
            tri.GetPointIds().SetId(1,r+numCuts)
            tri.GetPointIds().SetId(2,r+numCuts+1)
            triangleList.append(tri)
        
        triangles = vtk.vtkCellArray()
        for triangleItem in triangleList:
            triangles.InsertNextCell(triangleItem)
            
                
        grooveBoundaryAsPD= vtk.vtkPolyData()
        grooveBoundaryAsPD.SetPoints(pointsOnBoundaryOfGroove)  
        grooveBoundaryAsPD.SetPolys(triangles)        
        grooveBoundaryAsPD.Update()   
        
        grooveBoundaryAsPDList.append(grooveBoundaryAsPD)
         
        print "\n ^^^*********************************^ Len len(grooveTopSideProfiles))", len(grooveTopSideProfiles)
            
        
        LogVTK(grooveBoundaryAsPD,directoryForLog+"ptsboundinggroove_"+str(q-1)+"_.vtp")
        
#        filename=directoryForLog+"ptsboundinggroove_"+str(q-1)+"_.vtp"
#        writer_td90 = vtk.vtkXMLPolyDataWriter()
#        writer_td90.SetInputData(grooveBoundaryAsPD)
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td90.SetFileName(filename)
#        writer_td90.Write()
#        
#        
    
        
        
           
#            pointsForHull = vtk.vtkAppendPolyData()
#            pointsForHull.AddInput(grooveTopSideProfiles[q-1][1] )
#            pointsForHull.AddInput( grooveTopSideProfiles[q][0] )
#            pointsForHull.Update()
#            
#            print "\n # pts item left", q,grooveTopSideProfiles[q-1][1].GetNumberOfPoints()
#            print "\n # pts item right", q,grooveTopSideProfiles[q-1][1].GetNumberOfPoints()
#            print "\n # pts item appended", q,pointsForHull.GetOutput().GetNumberOfPoints()
#            
#            convhull=Compute2DHull(pointsForHull)
#            
#            writer_td90 = vtk.vtkXMLPolyDataWriter()
#            writer_td90.SetInputData(pointsForHull.GetOutput())
#            filename = "c:\\temp\\hullfortopofgroove"+"_"+ str(q) +"_.vtp"
#            #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#            writer_td90.SetFileName(filename)
#            writer_td90.Write()

    return(grooveBoundaryAsPDList)

def ProcessGrooveCovers(grooveBoundaryAsPDList, clippedOriginalMesh):
    
    # for each "groovecover"
    # 1) a1= compute 2D conver hull of groove cover and convert to surface
    # 2) b1 = clip "clipped tire mesh (After_clipped) to bounding box defined by convex hull, 
    # 3) c1 = resulting mesh of distance between a1 and b1
    # 4) d1= threshhold  on c1 to 2mm depth ()
    # 5) e1= march down d1, slice across and get max distList 
    # 6) f1= find best bit line using max dist list 
    # 7) g1= slice vertically down groove with a plane oriented according to best fit line 
    
    print "\n **************************** Entering ProcessGrooveCovers ************"
    print clippedOriginalMesh,"\n"
    grooveBoundaryAsPDList=[]


#        
    
    
    bbox=[0,0,0,0,0,0]
    grooveBoundaryAsHullList=[]
    origMeshClippedToGrooveList=[]
    

    for i,grooveBoundaryAsPD in enumerate(grooveBoundaryAsPDList):
        grooveBoundaryAsHull=Compute2DHull(grooveBoundaryAsPD)
        grooveBoundaryAsHullList.append(grooveBoundaryAsHull)
        
       
        
        bbox=grooveBoundaryAsHull.GetOutput().GetBounds()
        box=vtk.vtkBox()
        box.SetBounds(bbox[0],bbox[1],bbox[2],bbox[3],-1,1)        
        
        
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(clippedOriginalMesh.GetOutputPort())
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(1)
        clipper.Update()
        
        origMeshClippedToGroove = vtk.vtkDataSetSurfaceFilter()
        origMeshClippedToGroove.SetInputConnection(clipper.GetOutputPort())
        origMeshClippedToGroove.Update()
        origMeshClippedToGrooveList.append(origMeshClippedToGroove)
    
    
        
        LogVTK(origMeshClippedToGroove.GetOutput(), directoryForLog+"origmeshclippedtogroove_"+str(i)+"_.vtp")
        
    return(grooveBoundaryAsHullList,origMeshClippedToGrooveList)
    
def ProcessGrooveCoversDistance(grooveBoundaryAsPDList, clippedOriginalMesh):
    
    # for each "groovecover"
    # 1) a1= compute 2D conver hull of groove cover and convert to surface
    # 2) b1 = clip "clipped tire mesh (After_clipped) to bounding box defined by convex hull, 
    # 3) c1 = resulting mesh of distance between a1 and b1
    # 4) d1= threshhold  on c1 to 2mm depth ()
    # 5) e1= march down d1, slice across and get max distList 
    # 6) f1= find best bit line using max dist list 
    # 7) g1= slice vertically down groove with a plane oriented according to best fit line 
    
    print "\n **************************** Entering ProcessGrooveCovers ************"
    print clippedOriginalMesh,"\n"
    

    print "\n *99  Within  process groove mesh has these many points\n", clippedOriginalMesh.GetOutput().GetNumberOfPoints()
  

    
    LogVTK(clippedOriginalMesh.GetOutput(), directoryForLog+"clippedOriginalMeshInProcessGCDist"+".vtp")

# stub n 
#    grooveBoundaryAsPDList=[]
#    for i in range(numGrooves):
#        filename=stubDirectory+"ptsboundinggroove_"+str(i)+"_.vtp"
#        mesh=vtk.vtkXMLPolyDataReader()
#        mesh.SetFileName(filename)
#        mesh.Update()
#        grooveBoundaryAsPDList.append(mesh)
#    
#        
    
    print "\n  grooveBoundaryAsPDList, clippedOriginalMesh  ", len(grooveBoundaryAsPDList), clippedOriginalMesh.GetOutput().GetNumberOfPoints()

    grooveBoundaryAsHullList=[]
    origMeshClippedToGrooveList=[]
    

    for i,grooveBoundaryAsPD in enumerate(grooveBoundaryAsPDList):
        grooveBoundaryAsHull=Compute2DHull(grooveBoundaryAsPD)
        grooveBoundaryAsHullList.append(grooveBoundaryAsHull)
        
        
        bounds=grooveBoundaryAsHull.GetOutput().GetBounds()

        meshClippedToGroove=Clip(clippedOriginalMesh.GetOutput(),(bounds[0],bounds[1],bounds[2],bounds[3],-1,1) )
    
    
  
        
       
        origMeshClippedToGroove,avgDist=ComputeDistance(i,meshClippedToGroove,grooveBoundaryAsHull,0, True)
            
        
        LogVTK(origMeshClippedToGroove.GetOutput(), directoryForLog+"origmeshclippedtogroove_"+str(i)+"_.vtp")
        origMeshClippedToGrooveList.append(origMeshClippedToGroove)
        
    return(grooveBoundaryAsHullList,origMeshClippedToGrooveList)


def MergeGrooves(origMeshClippedToGrooveList):
    allmesh_clipped_to_grooves = vtk.vtkAppendPolyData()
    for origMeshClippedToGrooveListItem in origMeshClippedToGrooveList:       
        allmesh_clipped_to_grooves.AddInput(origMeshClippedToGrooveListItem.GetOutput())
        allmesh_clipped_to_grooves.Update()
    LogVTK(allmesh_clipped_to_grooves.GetOutput(), directoryForLog+"allmesh_limited_to_grooves.vtp")
    
def OrientPCA(grooveBoundaryAsHullList,origMeshClippedToGrooveList,clippedOrigMesh):
    
    allmesh_clipped_to_grooves = vtk.vtkAppendPolyData()
    for origMeshClippedToGrooveListItem in origMeshClippedToGrooveList:       
        allmesh_clipped_to_grooves.AddInput(origMeshClippedToGrooveListItem.GetOutput())
        allmesh_clipped_to_grooves.Update()
    LogVTK(allmesh_clipped_to_grooves.GetOutput(), directoryForLog+"allmesh_clipped_to_grooves.vtp")  
    orientationToXYZTransform=PcaGrooves(allmesh_clipped_to_grooves)
    TransformToFile(orientationToXYZTransform, directoryForLog+"orientationToXYZTransform.txt")
    
#    orientationToXYZTransform=vtk.vtkTransform()
#    orientationToXYZTransform.Identity()

    # create a new list of reoriented grooves
    allmesh_clipped_to_grooves_oriented=TransformPD(allmesh_clipped_to_grooves.GetOutput(),orientationToXYZTransform)
    grooveBoundaryAsHullOrientedList=[]
    origMeshClippedToGrooveOrientedList =[]
    
    for i,(origMeshClippedToGrooveListItem ,grooveBoundaryAsHullListItem) in enumerate(zip(origMeshClippedToGrooveList,grooveBoundaryAsHullList) ):
        origMeshClippedToGrooveOrientedList.append(TransformPD(origMeshClippedToGrooveListItem.GetOutput(),orientationToXYZTransform))
        grooveBoundaryAsHullOrientedList.append(TransformPD(grooveBoundaryAsHullListItem.GetOutput(),orientationToXYZTransform))
        LogVTK(TransformPD(origMeshClippedToGrooveListItem.GetOutput(),orientationToXYZTransform).GetOutput(),            directoryForLog+"origMeshClippedToGrooveListItemOriented"+str(i)+".vtp")   
        LogVTK(TransformPD(grooveBoundaryAsHullListItem.GetOutput(),orientationToXYZTransform).GetOutput(),            directoryForLog+"grooveBoundaryAsHullListItemOriented"+str(i)+".vtp")   
      
    
    LogVTK(allmesh_clipped_to_grooves_oriented.GetOutput(), directoryForLog+"allmesh_clipped_to_grooves_oriented_.vtp")  
    LogVTK(TransformPD(clippedOrigMesh.GetOutput(),orientationToXYZTransform).GetOutput(),directoryForLog+"After_clipped_oriented.vtp")   
    
    return(grooveBoundaryAsHullOrientedList, origMeshClippedToGrooveOrientedList,allmesh_clipped_to_grooves_oriented, orientationToXYZTransform)
        
#        filename=directoryForLog+"origmeshclippedtogroove_"+str(i)+"_.vtp"
#               
#        writer_td89 = vtk.vtkXMLPolyDataWriter()
#        writer_td89.SetInputData(origMeshClippedToGroove.GetOutput())
#      
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td89.SetFileName(filename)
#        writer_td89.Write()
#      
def FindBottomOfGroove(grooveBoundaryAsHullOrientedList,origMeshClippedToGrooveOrientedList,clippedOriginalMesh ):
    
    print "\n*********************** Entering FindBottomOfGroove ************************\n"
    
    for i,(origMeshClippedToGroove,grooveBoundaryAsHull) in enumerate(zip(origMeshClippedToGrooveOrientedList,grooveBoundaryAsHullOrientedList )): 
        deepPartOfGrooveMesh,avgDist=ComputeDistance(i,origMeshClippedToGroove,grooveBoundaryAsHull,0, False)
        maxDistPtDownGrooveList=FindMaxDistForGroove(deepPartOfGrooveMesh)
        bbox=deepPartOfGrooveMesh.GetSecondDistanceOutput().GetBounds()
        result=FindGrooveBestFitLine(str(i),maxDistPtDownGrooveList,bbox,clippedOriginalMesh)
        
        
              
        

        

def Compute2DHull(mesh):        
        
    ch = vtk.vtkDelaunay2D()
    ch.SetInputData(mesh)
    ch.Update()
     
    esch = vtk.vtkDataSetSurfaceFilter()
    esch.SetInputConnection(ch.GetOutputPort())
    esch.Update()
    
    return(esch)
    
            
    



def BorderForBlocks(boxList):
   
#    foo=[]
#    foo.append(boxList[0])
#    boxList=foo
    for i,boxListItem in enumerate(boxList):
        print "\n &&&&&&&&&&&&&&&& i", i
        ch = vtk.vtkDelaunay2D()
        ch.SetInputConnection(boxListItem.GetOutputPort())
        ch.Update()
         
        esch = vtk.vtkDataSetSurfaceFilter()
        esch.SetInputConnection(ch.GetOutputPort())
        esch.Update()
        
                # # Extract boundary edges
        boundary = vtk.vtkFeatureEdges()
        boundary.BoundaryEdgesOn()
        boundary.FeatureEdgesOff()
        boundary.ManifoldEdgesOff()
        boundary.NonManifoldEdgesOff()
        boundary.SetInputConnection(esch.GetOutputPort())
        boundary.Update()
    
        
        bbox=boundary.GetOutput().GetBounds()
        print "\n bbox for boundary", bbox
        
       
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(boundary.GetOutput())
        locator.BuildLocator()
        
     
        pbl=locator.FindClosestPoint((bbox[0],bbox[2],bbox[4]) )
        
   
        ptl=locator.FindClosestPoint((bbox[1],bbox[2],bbox[4]) )
        pbr=locator.FindClosestPoint((bbox[0],bbox[3],bbox[4]) )
        ptr=locator.FindClosestPoint((bbox[1],bbox[3],bbox[4]) )
        
        cornerPts= [pbl,ptl,pbr,ptr]
        
        cornerCells=[]
        
        
        print "\n Corner Pts\n", cornerPts          
        
        # find cells associated with each point
        for k,cornerPtItem in enumerate(cornerPts):
            coords=[0.0,0.0,0.0]
            boundary.GetOutput().GetPoint(cornerPtItem,coords)
            locatorc = vtk.vtkCellLocator()
            locatorc.SetDataSet(boundary.GetOutput())
            locatorc.BuildLocator()
        
            cellID = locatorc.FindCell(coords) 
            print "\n closest cell for box corner", k, "is", cellID
            cornerCells.append(cellID)

        
        ptsToRemove=[pbl,ptl,pbr,ptr]
        
       
        ptsToRemoveVTKList = vtk.vtkIntArray()
        ptsToRemoveVTKList.SetNumberOfComponents(1)
        ptsToRemoveVTKList.SetName("ptsToRemoveVTKList")
        
    
        #ptsInHullRevVTKList= vtk.vtkIdList()
        for j,ptToRemove in enumerate(ptsToRemove):
            ptsToRemoveVTKList.InsertValue(j,ptToRemove)
            
            
        cellsToRemove=cornerCells    
        
        cellsToRemoveVTKList = vtk.vtkIntArray()
        cellsToRemoveVTKList.SetNumberOfComponents(1)
        cellsToRemoveVTKList.SetName("cellsToRemoveVTKList")
        
    
        #cellsInHullRevVTKList= vtk.vtkIdList()
        for j,cellToRemove in enumerate(cellsToRemove):
            cellsToRemoveVTKList.InsertValue(j,cellToRemove)
            
        
            #x=ptsInHullRevVTKList.InsertUniqueId(ptInHullReviewed)


        print "\n Ptsa to remove list ", ptsToRemove
        #select the hull pts and export them
        selectionNode=vtk.vtkSelectionNode()
        #selectionNode.SetFieldType(1)
        selectionNode.SetFieldType(0)
        selectionNode.SetContentType(4)
      
        selectionNode.GetProperties().Set(selectionNode.INVERSE(),1)
        #selectionNode.SetSelectionList(ptsToRemove)
#        selectionNode.SetFieldType(vtkSelectionNode::POINT)
#        selectionNode.SetContentType(vtkSelectionNode::INDICES)
        selectionNode.SetSelectionList(cellsToRemoveVTKList)

        selection =vtk.vtkSelection()
        selection.AddNode(selectionNode)
        selection.Update()
        

        extractSelection =vtk.vtkExtractSelection()
#    extractSelection.SetInputConnection(0, pointSource.GetOutputPort())
        extractSelection.SetInputData(0, boundary.GetOutput() )
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        esx4 = vtk.vtkDataSetSurfaceFilter()
        esx4.SetInputConnection(extractSelection.GetOutputPort())
        esx4.Update()


        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputConnection(esx4.GetOutputPort() )
        
        connectivityFilter.SetExtractionModeToClosestPointRegion ()
        
        
        print "\n bbox", bbox[0], bbox[1]
        x1=bbox[0]
        y1=(bbox[2]+bbox[3])/2.0
        z1=bbox[4]
        #a1=[bbox[0],(bbox[2]+bbox[3])/2  ,  bbox(4) ]
        #connectivityFilter.SetClosestPoint((bbox[0],(bbox[2]+bbox[3])/2  ,  bbox(4)  ) )
        connectivityFilter.SetClosestPoint(x1,y1,z1 )
        connectivityFilter.ColorRegionsOn()
       
        connectivityFilter.Update()
        
        LogVTK(connectivityFilter.GetOutput(),directoryForLog+"block2dhullminuscorners_leftside"+str(i)+".vtp")
#             
#        writer_td89 = vtk.vtkXMLPolyDataWriter()
#        writer_td89.SetInputData(connectivityFilter.GetOutput())
#        filename = directoryForLog+"block2dhullminuscorners_leftside"+str(i)+".vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td89.SetFileName(filename)
#        writer_td89.Write()
#        
        x1=bbox[1]
        y1=(bbox[2]+bbox[3])/2.0
        z1=bbox[4]
    
        connectivityFilter.SetClosestPoint(x1,y1,z1 )
        connectivityFilter.ColorRegionsOn()
        
        connectivityFilter.ColorRegionsOn()
        connectivityFilter.Update()
        
        LogVTK(connectivityFilter.GetOutput(), directoryForLog+"block2dhullminuscorners_rightside"+str(i)+".vtp")

#        
#        writer_td90 = vtk.vtkXMLPolyDataWriter()
#        writer_td90.SetInputData(connectivityFilter.GetOutput())
#        filename = directoryForLog+"block2dhullminuscorners_rightside"+str(i)+".vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td90.SetFileName(filename)
#        writer_td90.Write()

       
        # find the cells closest to the four corners of the bounding box
        print "\nClosest point is ",pbl
        print "\n num pts boundary", boundary.GetOutput().GetNumberOfPoints()
        
        LogVTK(boundary.GetOutput(),directoryForLog+"block2dhull_"+str(i)+".vtp")
        
#        writer_td88 = vtk.vtkXMLPolyDataWriter()
#        writer_td88.SetInputData(boundary.GetOutput())
#        filename = directoryForLog+"block2dhull_"+str(i)+".vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td88.SetFileName(filename)
#        writer_td88.Write()
#        
               # find the cells closest to the four corners of the bounding box
        print "\nClosest point is ",pbl
        
        LogVTK(esx4.GetOutput(), directoryForLog+"block2dhullminuscorners_"+str(i)+".vtp")        
        
#        writer_td87 = vtk.vtkXMLPolyDataWriter()
#        writer_td87.SetInputData(esx4.GetOutput())
#        filename = directoryForLog+"block2dhullminuscorners_"+str(i)+".vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td87.SetFileName(filename)
#        writer_td87.Write()
#        
#   
#        
        
        

    
def FormBalancedMeshFromBlockMask(gp,meshNoGrooves):
    numBlocks = len(gp)+1
    bbox=[0,0,0,0,0,0]
    bbox=meshNoGrooves.GetOutput().GetBounds()
    
        
    print "\n ******************* Entering  FormBalancedMeshFromBlockMask"
    print "\nnumber of blocks ", numBlocks    
    
    boxList=[]    
    for i in range(numBlocks):
        print "\n working on block", i
        if (i==0):
            bounds=[bbox[0],gp[0][0],bbox[2],bbox[3],bbox[4],bbox[5]]
        else:
            if (i==numBlocks-1):
                 bounds=[gp[i-1][1],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5]]
            else:
                 bounds=[gp[i-1][1],gp[i][0],bbox[2],bbox[3],bbox[4],bbox[5]]
                
        box=vtk.vtkBox()
        box.SetBounds(bounds)
        
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(meshNoGrooves.GetOutputPort())
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(1)
        clipper.Update()
        
        es1 = vtk.vtkDataSetSurfaceFilter()
        es1.SetInputConnection(clipper.GetOutputPort())
        es1.Update()
        
           
        boxList.append(es1)


    print "\n*********** box list\n", boxList
    
    return(boxList)
  
#    borderList = GetProfileTreadBlocks(boxList)
  
     
        
        
  
        

#
def  FindMaxDistForGroove(grooveNumber,pd):
   
    
    print "\n&&&&&&&&&&&&&&&&&&& Entering FindMaxDistForGroove "   
    #change
    #print "number of pts", mesh.GetSecondDistanceOutput().GetNumberOfPoints()
    print "number of pts", pd.GetNumberOfPoints()  
    
    #LogVTK(mesh.GetSecondDistanceOutput(), directoryForLog+"fmdfg_mesh" + str(i)+"_.vtp")
    LogVTK(pd, directoryForLog+"fmdfg_mesh" + str(grooveNumber)+"_.vtp")
    
#    
#    writer_cutter1 = vtk.vtkXMLPolyDataWriter()
#    writer_cutter1.SetInputData(mesh.GetSecondDistanceOutput())
#    fname=directoryForLog+"fmdfg_mesh.vtp"
#    writer_cutter1.SetFileName(fname)
#    writer_cutter1.Write()
#        

  
  
#  
#    bbox=[0,0,0,0,0,0]


    

    
    fname=directoryForLog+"grMaxDistPts_"+str(grooveNumber)+".txt"
    f = open(fname, 'w+')
    f.write("X,Y,Z\n" )
    
    bbox=pd.GetBounds()
    numcuts=20
    maxYForCuts=max( ( bbox[3]-bbox[2] ), ( bbox[3]-bbox[2] ) )
    minYGrooveA=bbox[2]
    yInterval=maxYForCuts/float(numcuts)
    
#   
##    bbox=mesh.GetOutput().GetBounds()
#    numcuts=100
##    maxYForCuts=max( ( bbox[3]-bbox[2] ), ( bbox[3]-bbox[2] ) )
#    minYGrooveA=0.02
#    yInterval=0.02
    
    
    minNumberPtsInGoodCut=5
    

    #fname=directoryForLog+"gr_GrooveA.vtp"


       
#    planeA=vtk.vtkPlane()
#    planeA.SetOrigin( (bbox[0]+bbox[1])/2     ,  0 ,0)
#    planeA.SetNormal(0,1,0)
#    cutterA=vtk.vtkCutter()
#    cutterA.SetCutFunction(planeA)
#    cutterA.SetInputConnection(mesh.GetOutputPort())
#    cutterA.Update()
    

    
#    writer_cutter1q = vtk.vtkXMLPolyDataWriter()
#    writer_cutter1q.SetInputData(cutterA.GetOutput())
#    fname="c:\\temp\\cutterA"+strid+".vtp"
#    writer_cutter1q.SetFileName(fname)
#    writer_cutter1q.Write()
    
    ptMaxList=[]
    for i in range(0,numcuts):
        
#        print "\n ### cut # ", i
#        print "\n A cut pt y", minYGrooveA+yInterval*float(i)

            
        planeA=vtk.vtkPlane()
        planeA.SetOrigin(0,minYGrooveA+yInterval*float(i),0)
        planeA.SetNormal(0,1,0)
        
        cutterA=vtk.vtkCutter()
        cutterA.SetCutFunction(planeA)
        #change
        #cutterA.SetInputData(mesh.GetSecondDistanceOutput())
        cutterA.SetInputData(pd)
        cutterA.Update()
        
        
        
        distListArrayA = cutterA.GetOutput().GetPointData().GetArray(3)

        LogVTK(cutterA.GetOutput(), directoryForLog+"grvt"+str(grooveNumber)+"_"+str(i)+".vtp")
        
#        writer_cutter1 = vtk.vtkXMLPolyDataWriter()
#        writer_cutter1.SetInputData(cutterA.GetOutput())
#        fname=directoryForLog+"grvt"+str(i)+".vtp"
#        writer_cutter1.SetFileName(fname)
#        writer_cutter1.Write()
#        
        
        pt =[0,0,0]
       
        print "\n # of PTs", cutterA.GetOutput().GetNumberOfPoints(),"\n"
        if (minNumberPtsInGoodCut>cutterA.GetOutput().GetNumberOfPoints() ):
            continue
       
       
       
       
       
        maxDist=-999
        maxDistPtA=0
        
       
        for j1 in range(0,cutterA.GetOutput().GetNumberOfPoints()):
                distA=distListArrayA.GetTuple(j1)[0]
                #print "\n ********** Distance is ", distA, minDist
                if (maxDist<distA):
                    maxDist=distA
                    pt = cutterA.GetOutput().GetPoint(i)
                    maxDistPtA=(pt[0],pt[1],pt[2],maxDist)
        
        print "\n max dist point for cut # ", i, "   coords", maxDistPtA, "Distance", maxDist      
        
        ptMaxList.append(maxDistPtA)
#        
#        line= str(minDistPtA[0])+","+str(minDistPtA[1])+","+str(minDistPtA[2]) +","+str(minDist)+"\n"
#        f.write( line )
            

        print "\n *************** Groove Max List *************", ptMaxList
       
       

#    
    f.close()    
    
    return(ptMaxList)  
    
    
    


def GetGrooveCenterPolylines(filename):
    
    print "\n ************ Entering GetGrooveCenterPolylines"
    
    mesh=vtk.vtkXMLPolyDataReader()
    mesh.SetFileName(filename)
    mesh.Update()
    
    thresholdDist = vtk.vtkThreshold()
    thresholdDist.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
    
    thresholdDist.SetInputData(mesh.GetOutput())
    thresholdDist.ThresholdBetween(-1,-0.001)

   
    thresholdDist.Update()
    
     
    es_thresholdDist = vtk.vtkDataSetSurfaceFilter()
    es_thresholdDist.SetInputConnection(thresholdDist.GetOutputPort())
    es_thresholdDist.Update()
    
    #numberOfPointArrays=reader.GetOutput().GetPointData().GetNumberOfArrays()
    #print "\n number of Point Arrays\n", numberOfPointArrays
    #
    #    
    #distPts = reader.GetOutput().GetPointData().GetArray(1)
    
    numPts =  es_thresholdDist.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es_thresholdDist.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    

    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
            
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)
   
    print "\n** sorted by second **\n"
    
   
    #print  sorted_by_cf_regionsize[int(ner)-1]
    #print  sorted_by_cf_regionsize[int(ner)-2]
    for j in range(0,ner):
        print  sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    connectivityFilter.Update()
    
    
       
   
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
  

    print "\n number of regions to isolate\n ", int(ner)
    
    LogVTK(connectivityFilter.GetOutput(), directoryForLog+"regions_before_added.vtp")
#    writer_td52 = vtk.vtkXMLPolyDataWriter()
#    writer_td52.SetInputData(connectivityFilter.GetOutput())
#    writer_td52.SetFileName(directoryForLog+"regions_before_added.vtp")
#    writer_td52.Write()

    
    #xyRatioThresh=0.3
    xyRatioThresh=0.25
    saBBRatioRatioThresh=0.25
    saThresh=0.00005
    
    numGrooves=0
    
    for j in range(ner):
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]

        if (j<>0):
            connectivityFilter.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
            #connectivityFilter.Update()
        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        #connectivityFilter.Update()
        
        cf = vtk.vtkCleanPolyData()
        cf.SetInputConnection(connectivityFilter.GetOutputPort())
        cf.Update()
            
        bbOneRegion=[0,0,0,0,0,0]
        bbOneRegion=cf.GetOutput().GetBounds()
        print "\n bounds of current region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
        print "\n bounds of current region\n", abs(bbOneRegion[0]-bbOneRegion[1]), abs(bbOneRegion[2]-bbOneRegion[3]),abs(bbOneRegion[4]-bbOneRegion[5])
        xyRatio=abs(bbOneRegion[0]-bbOneRegion[1])/abs(bbOneRegion[2]-bbOneRegion[3])
        massProperty = vtk.vtkMassProperties()
        massProperty.SetInputConnection(cf.GetOutputPort())
        massProperty.Update()
        sa = massProperty.GetSurfaceArea()
        print "\nSurface Area", sa
        saBBRatio = sa/ ( (bbOneRegion[1]-bbOneRegion[0]) * (bbOneRegion[3]-bbOneRegion[2])  )
        print "\nBBArea", (bbOneRegion[1]-bbOneRegion[0]) * (bbOneRegion[3]-bbOneRegion[2]) 
        print "\nsaBBRatiio", saBBRatio
        
        if (sa> 0.0001):
            
            LogVTK(cf.GetOutput(), directoryForLog+"isolatingGroove_"+str(j)+".vtp")

#            writer_td51 = vtk.vtkXMLPolyDataWriter()
#            #writer = vtk.vtkSimplePointsWriter()
#            filename=directoryForLog+"isolatingGroove_"+str(j)+".vtp"
#            writer_td51.SetInputData(cf.GetOutput())
#            writer_td51.SetFileName(filename)
#            writer_td51.Write()
#            numGrooves=numGrooves+1
    
            grooveCenterLine=   FindMaxDistForGroove  (cf)    
            
        else:
            break

        
        numPts =  cf.GetOutput().GetNumberOfPoints()
        print "\nnum pts after clean", numPts
        


def GetArrayByNamePC(pd,name,pcFlag) :

    
    if (pcFlag):
        pdData=pd.GetPointData()
    else:
        pdData=pd.GetCellData()
        
    numberOfArrays = pdData.GetNumberOfArrays()
        
  
        
    #print "\n # of arrays ", numberOfPointArrays,"\n"
    for q in range(numberOfArrays):
        aname = pdData.GetArrayName(q)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", q , aname   
        if (aname==name):
            break
    return(q)       
       
        
def GetArrayByName(pd,name)   :
    
    numberOfPointArrays = pd.GetPointData().GetNumberOfArrays()
    #print "\n # of arrays ", numberOfPointArrays,"\n"
    for q in range(numberOfPointArrays):
        aname = pd.GetPointData().GetArrayName(q)
        #print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", q , aname   
        if (aname==name):
            break
    return(q)



def ComputeDistance1(i,mesh1,mesh2,threshStartDist,computeAverage):
    
    print "\n************** Entering ComputeDistance ONE *************88"
    print "\n directory for log", directoryForLog, "\n"
    
    print "\n && i = ", i
    
    LogVTK(mesh1.GetOutput(), directoryForLog+"mesh1rfts"+str(i)+"_.vtp")
    LogVTK(mesh2.GetOutput(), directoryForLog+"mesh2rf0"+str(i)+"_.vtp")
    
   
    
    # clip mesh 2 to bounding box of mesh 2
#    bounds=mesh2.GetOutput().GetBounds()
#    print "\nbounds are ", bounds
#    box=vtk.vtkBox()
#    box.SetBounds(bounds)
#    
    #mesh2Clipped=Clip(mesh1.GetOutput(),bounds)
#    mesh1Clipped=Clip(mesh1.GetOutput(),(bounds[0],bounds[1],bounds[2],bounds[3],-1,1) )
#    
    LogVTK(mesh1.GetOutput(), directoryForLog+"mesh1"+str(i)+"_.vtp")
  
#    
#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"compdistmesh_"+str(i)+"_.vtp"
#    writer_td.SetInputData(mesh1.GetOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()

    # remove everything but top cells from convex hull
    
    treadDepth = vtk.vtkDistancePolyDataFilter()
    treadDepth.SetInputConnection(0,mesh1.GetOutputPort())
    treadDepth.SetInputConnection(1,mesh2.GetOutputPort())
    treadDepth.ComputeSecondDistanceOn()
    #treadDepth.SignedDistanceOff()
    treadDepth.Update()
    
    numberOfPointArrays = treadDepth.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for q in range(numberOfPointArrays):
        aname = treadDepth.GetOutput().GetPointData().GetArrayName(q)
        print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", q , aname    
    
    
    LogVTK( treadDepth.GetSecondDistanceOutput(), directoryForLog+"treaddepth2ndoutput_"+str(i)+"_.vtp")
    LogVTK( treadDepth.GetOutput(), directoryForLog+"treaddepthoutput_"+str(i)+"_.vtp")
   
#    distCells = treadDepth.GetOutput().GetCellData().GetArray(0)
#    ##areaCells=normals1.GetOutput().GetCellData().GetArray(0)
#    distPts = treadDepth.GetOutput().GetPointData().GetArray(1)
#    
#    numCells =  treadDepth.GetOutput().GetNumberOfCells()
#    numPts =  treadDepth.GetOutput().GetNumberOfPoints()
#    data = treadDepth.GetOutput()
    
    
    

    if (threshStartDist<>0):
        thresholdDist = vtk.vtkThreshold()
        thresholdDist.SetInputData(treadDepth.GetOutput())
        thresholdDist.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
        thresholdDist.ThresholdBetween(-threshStartDist,threshStartDist)
        thresholdDist.Update()
    
    
        esDist = vtk.vtkDataSetSurfaceFilter()
        esDist.SetInputConnection(thresholdDist.GetOutputPort())
        esDist.Update()
        
        treadDepth=esDist
        LogVTK( treadDepth.GetOutput(), directoryForLog+"Afterthres_treaddepthoutput_"+str(i)+"_.vtp")

    
    
    #changed!
    #treadDepth.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
    index=GetArrayByName(treadDepth.GetOutput(),"Distance")
    #distPts = treadDepth.GetOutput().GetPointData().GetArrayByName("Distance")
    distPts = treadDepth.GetOutput().GetPointData().GetArray(index)
    
    print "\n Dist pts", treadDepth.GetOutput().GetNumberOfPoints()
    
    avgDist=0
    if (computeAverage):
    
        totalDistance=0
        for i in range(treadDepth.GetOutput().GetNumberOfPoints()):
          
            disttup=distPts.GetTuple(i)
            #print "\n &&&&&&&&&&&", disttup
            dist = abs(disttup[0])
            totalDistance=totalDistance+dist
        
            
        avgDist = float(totalDistance)/float(treadDepth.GetOutput().GetNumberOfPoints())
        
        print "\n averagge dist is ", avgDist
    

#    treadDepth.Update()
    
    print "\n ********** before exiting distance calc ****************\n"
    
    print "\nnumber of pts", mesh1.GetOutput().GetNumberOfPoints()
    print "\nnumber of pts", mesh2.GetOutput().GetNumberOfPoints()
    print "\nnumber of pts", treadDepth.GetOutput().GetNumberOfPoints()
 

#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"treaddepth_"+str(i)+"_.vtp"
#    writer_td.SetInputData(treadDepth.GetSecondDistanceOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()
    
    return (treadDepth,avgDist)





def ComputeDistance(i,mesh1,mesh2,threshStartDist,computeAverage):
    
    print "\n************** Entering ComputeDistance *********ONE ****"
    
    print "\n && i = ", i
    
    LogVTK(mesh1.GetOutput(), directoryForLog+"mesh1"+str(i)+"_.vtp")
    LogVTK(mesh2.GetOutput(), directoryForLog+"mesh2"+str(i)+"_.vtp")
    
   
    
    # clip mesh 2 to bounding box of mesh 2
#    bounds=mesh2.GetOutput().GetBounds()
#    print "\nbounds are ", bounds
#    box=vtk.vtkBox()
#    box.SetBounds(bounds)
#    
    #mesh2Clipped=Clip(mesh1.GetOutput(),bounds)
#    mesh1Clipped=Clip(mesh1.GetOutput(),(bounds[0],bounds[1],bounds[2],bounds[3],-1,1) )
#    
    LogVTK(mesh1.GetOutput(), directoryForLog+"mesh1"+str(i)+"_.vtp")
  
#    
#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"compdistmesh_"+str(i)+"_.vtp"
#    writer_td.SetInputData(mesh1.GetOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()

    # remove everything but top cells from convex hull
    
    treadDepth = vtk.vtkDistancePolyDataFilter()
    treadDepth.SetInputConnection(0,mesh1.GetOutputPort())
    treadDepth.SetInputConnection(1,mesh2.GetOutputPort())
    treadDepth.ComputeSecondDistanceOn()
    #treadDepth.SignedDistanceOff()
    treadDepth.Update()
    
    numberOfPointArrays = treadDepth.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n # of arrays ", numberOfPointArrays,"\n"
    for q in range(numberOfPointArrays):
        aname = treadDepth.GetOutput().GetPointData().GetArrayName(q)
        print "\n ++++++++++++++++++++++++++++++ Array Name for pd ", q , aname    
    
    
    LogVTK( treadDepth.GetSecondDistanceOutput(), directoryForLog+"treaddepth2ndoutput_"+str(i)+"_.vtp")
    LogVTK( treadDepth.GetOutput(), directoryForLog+"treaddepthoutput_"+str(i)+"_.vtp")
   
#    distCells = treadDepth.GetOutput().GetCellData().GetArray(0)
#    ##areaCells=normals1.GetOutput().GetCellData().GetArray(0)
#    distPts = treadDepth.GetOutput().GetPointData().GetArray(1)
#    
#    numCells =  treadDepth.GetOutput().GetNumberOfCells()
#    numPts =  treadDepth.GetOutput().GetNumberOfPoints()
#    data = treadDepth.GetOutput()
    
    
    

    if (threshStartDist<>0):
        thresholdDist = vtk.vtkThreshold()
        thresholdDist.SetInputData(treadDepth.GetOutput())
        thresholdDist.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
        thresholdDist.ThresholdBetween(threshStartDist,1)
        thresholdDist.Update()
    
    
        esDist = vtk.vtkDataSetSurfaceFilter()
        esDist.SetInputConnection(thresholdDist.GetOutputPort())
        esDist.Update()
        
        treadDepth=esDist
        LogVTK( treadDepth.GetOutput(), directoryForLog+"Afterthres_treaddepthoutput_"+str(i)+"_.vtp")

    
    
    #changed!
    #treadDepth.SetInputArrayToProcess(0, 0, 0, 1, "Distance")
    index=GetArrayByName(treadDepth.GetOutput(),"Distance")
    #distPts = treadDepth.GetOutput().GetPointData().GetArrayByName("Distance")
    distPts = treadDepth.GetOutput().GetPointData().GetArray(index)
    
    print "\n Dist pts", treadDepth.GetOutput().GetNumberOfPoints()
    
    avgDist=0
    if (computeAverage):
    
        totalDistance=0
        for i in range(treadDepth.GetOutput().GetNumberOfPoints()):
          
            disttup=distPts.GetTuple(i)
            #print "\n &&&&&&&&&&&", disttup
            dist = abs(disttup[0])
            totalDistance=totalDistance+dist
        
            
        avgDist = float(totalDistance)/float(treadDepth.GetOutput().GetNumberOfPoints())
        
        print "\n averagge dist is ", avgDist
    

#    treadDepth.Update()
    
    print "\n ********** before exiting distance calc ****************\n"
    
    print "\nnumber of pts", mesh1.GetOutput().GetNumberOfPoints()
    print "\nnumber of pts", mesh2.GetOutput().GetNumberOfPoints()
    print "\nnumber of pts", treadDepth.GetOutput().GetNumberOfPoints()
 

#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"treaddepth_"+str(i)+"_.vtp"
#    writer_td.SetInputData(treadDepth.GetSecondDistanceOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()
    
    return (treadDepth,avgDist)
    


def ComputeDistanceNew(i,pd1,pd2):
    
    #print "\n************** Entering ComputeDistance "
    
    #LogVTK(pd1, directoryForLog+"pd2"+str(i)+"_.vtp")
    #LogVTK(pd2, directoryForLog+"pd2"+str(i)+"_.vtp")
    
    
    # clip mesh 2 to bounding box of mesh 2
#    bounds=mesh2.GetOutput().GetBounds()
#    print "\nbounds are ", bounds
#    box=vtk.vtkBox()
#    box.SetBounds(bounds)
#    
    #mesh2Clipped=Clip(mesh1.GetOutput(),bounds)
#    mesh1Clipped=Clip(mesh1.GetOutput(),(bounds[0],bounds[1],bounds[2],bounds[3],-1,1) )
#    
    #LogVTK(pd1, directoryForLog+"mesh1"+str(i)+"_.vtp")
  
#    
#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"compdistmesh_"+str(i)+"_.vtp"
#    writer_td.SetInputData(mesh1.GetOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()

    # remove everything but top cells from convex hull
    
    treadDepth = vtk.vtkDistancePolyDataFilter()
    treadDepth.SetInputData(0,pd1)
    treadDepth.SetInputData(1,pd2)
    treadDepth.SignedDistanceOff	()	

    treadDepth.ComputeSecondDistanceOn()
    #treadDepth.SignedDistanceOff()
    treadDepth.Update()
    
    #LogVTK( treadDepth.GetSecondDistanceOutput(), directoryForLog+"treaddepth2ndoutput_"+str(i)+"_.vtp")
    #LogVTK( treadDepth.GetOutput(), directoryForLog+"treaddepthoutput_"+str(i)+"_.vtp")
   
#    distCells = treadDepth.GetOutput().GetCellData().GetArray(0)
#    ##areaCells=normals1.GetOutput().GetCellData().GetArray(0)
#    distPts = treadDepth.GetOutput().GetPointData().GetArray(1)
#    
#    numCells =  treadDepth.GetOutput().GetNumberOfCells()
#    numPts =  treadDepth.GetOutput().GetNumberOfPoints()
#    data = treadDepth.GetOutput()
    
    
    


#    treadDepth.Update()
    
    #print "\n ********** before exiting distance calc ****************\n"
    
    #print "\nnumber of pts", pd1.GetNumberOfPoints()
    #print "\nnumber of pts",pd2.GetNumberOfPoints()
    #print "\nnumber of pts", treadDepth.GetOutput().GetNumberOfPoints()
 

#    writer_td = vtk.vtkXMLPolyDataWriter()
#    filename=directoryForLog+"treaddepth_"+str(i)+"_.vtp"
#    writer_td.SetInputData(treadDepth.GetSecondDistanceOutput())
#    writer_td.SetFileName(filename)
#    writer_td.Write()
    
    return (treadDepth)
    

def ContoursByTreadDepth(tireState):
    contours = vtk.vtkContourFilter()
    contours.SetInputData(tireState[3][0].GetOutputPort())
    contours.SetNumberOfContours(10)
#contours.GenerateValues(10, 0, 256)
#    contours.SetValue(0, 30)
#    contours.SetValue(1, 100)
#    contours.SetValue(2, 150)
#    contours.SetValue(3, 220)

# returns the 'top' part of a convex hull to be used for distance calculation
def TopOfConvexHull(pd):
    
    # process the clipped tread
  
    
    ch = vtk.vtkDelaunay3D()
    ch.SetInputData(pd)
    ch.Update()
     
    esch = vtk.vtkDataSetSurfaceFilter()
    esch.SetInputConnection(ch.GetOutputPort())
    esch.Update()
    eschMapper = vtk.vtkPolyDataMapper()
    eschMapper.SetInputConnection(esch.GetOutputPort())
    eschActor = vtk.vtkActor()
    eschActor.SetMapper(eschMapper)
    
    sdf=vtk.vtkLinearSubdivisionFilter()
    sdf.SetInputConnection(esch.GetOutputPort())
    sdf.SetNumberOfSubdivisions(3)
    sdf.Update()


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputConnection(sdf.GetOutputPort())
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays
    #for i in range(0,numberOfCellArrays):
    #        print  tireDelNormals.GetOutput().GetCellData().GetArray(i).GetName()


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()
    data = normalsch.GetOutput()

    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    LogVTK( normalsch.GetOutput(), directoryForLog+"rawconvexhullfilterbeforethresh.vtp")

    
    

    
    
#    thresholdx = vtk.vtkThreshold()
#    thresholdx.SetInputData(normalsch.GetOutput())
#    #thresholdx.SetAttributeModeToUseCellData()
#    #thresholdx.ThresholdBetween(-0.95,0.95)
#    thresholdx.ThresholdBetween(0.95,1)
#    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "zn")
#    thresholdx.Update()
#
#
#    esx2 = vtk.vtkDataSetSurfaceFilter()
#    esx2.SetInputConnection(thresholdx.GetOutputPort())
#    esx2.Update()
#  
#    LogVTK( esx2.GetOutput(), directoryForLog+"convexhullfilter.vtp")
##    writer_td = vtk.vtkXMLPolyDataWriter()
##    #writer = vtk.vtkSimplePointsWriter()
##    writer_td.SetInputData(esx2.GetOutput())
##    writer_td.SetFileName(directoryForLog+"convexhullfilter.vtp")
##    writer_td.Write()
##    
#    return(esx2.GetOutput())
    
     
    thresholdy = vtk.vtkThreshold()
    thresholdy.SetInputData(normalsch.GetOutput())
   
    thresholdy.ThresholdBetween(-0.4,0.4)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.Update()
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(thresholdy.GetOutput())
   
    thresholdx.ThresholdBetween(-0.4,0.4)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdx.GetOutput())
    #thresholdx.SetAttributeModeToUseCellData()
    #thresholdx.ThresholdBetween(-0.95,0.95)
    thresholdz.ThresholdBetween(-1,0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    
    


    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdz.GetOutputPort())
    esz2.Update()
  
    LogVTK( esz2.GetOutput(), directoryForLog+"convexhullfilter.vtp")
#    writer_td = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    writer_td.SetInputData(esx2.GetOutput())
#    writer_td.SetFileName(directoryForLog+"convexhullfilter.vtp")
#    writer_td.Write()
#    
    return(esz2)
    
# returns the 'top' part of a convex hull to be used for distance calculation
def TopOfConvexHull2(pd):
    
    # process the clipped tread
  
    
#    ch = vtk.vtkDelaunay3D()
#    ch.SetInputData(pd)
#    ch.Update()
#    

     
#    esch = vtk.vtkDataSetSurfaceFilter()
#    esch.SetInputConnection(ch.GetOutputPort())
#    esch.Update()
 
    chPD=ConvexHullSciPy(pd)
     
    
    sdf=vtk.vtkLinearSubdivisionFilter()
    sdf.SetInputData(chPD)
    sdf.SetNumberOfSubdivisions(3)
    sdf.Update()


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputConnection(sdf.GetOutputPort())
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays
    #for i in range(0,numberOfCellArrays):
    #        print  tireDelNormals.GetOutput().GetCellData().GetArray(i).GetName()


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()


    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    LogVTK( normalsch.GetOutput(), directoryForLog+"rawconvexhullfilterbeforethresh.vtp")

    

    

#    
    return(normalsch.GetOutput())
    

# returns the 'top' part of a convex hull to be used for distance calculation
def TopOfConvexHull1(pd):
    
    # process the clipped tread
  
    
    ch = vtk.vtkDelaunay3D()
    ch.SetInputData(pd)
    ch.SetOffset(10)
    ch.Update()
    

     
    esch = vtk.vtkDataSetSurfaceFilter()
    esch.SetInputConnection(ch.GetOutputPort())
    esch.Update()
    eschMapper = vtk.vtkPolyDataMapper()
    eschMapper.SetInputConnection(esch.GetOutputPort())
    eschActor = vtk.vtkActor()
    eschActor.SetMapper(eschMapper)
    
    sdf=vtk.vtkLinearSubdivisionFilter()
    sdf.SetInputConnection(esch.GetOutputPort())
    sdf.SetNumberOfSubdivisions(3)
    sdf.Update()


    normalsch =vtk.vtkPolyDataNormals ()
    normalsch.SetInputConnection(sdf.GetOutputPort())
    normalsch.SetComputeCellNormals (1) 
    normalsch.AutoOrientNormalsOn()
    normalsch.Update()
    
    
    numberOfCellArrays=normalsch.GetOutput().GetPointData().GetNumberOfArrays()
    print "\n number of Cell Arrays\n", numberOfCellArrays
    #for i in range(0,numberOfCellArrays):
    #        print  tireDelNormals.GetOutput().GetCellData().GetArray(i).GetName()


    normCells = normalsch.GetOutput().GetCellData().GetNormals()
    areaCells=normalsch.GetOutput().GetCellData().GetArray(0)

    numCells =  normalsch.GetOutput().GetNumberOfCells()
    data = normalsch.GetOutput()

    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    itemList=[]
    for i in range (0,numCells):

        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)


    itemList.sort(key=lambda tup: tup[4])


    normalsch.GetOutput().GetCellData().AddArray(data1)
    normalsch.GetOutput().GetCellData().AddArray(data2)
    normalsch.GetOutput().GetCellData().AddArray(data3)
    normalsch.Update()
    
    LogVTK( normalsch.GetOutput(), directoryForLog+"rawconvexhullfilterbeforethresh.vtp")

    
    

    
    
#    thresholdx = vtk.vtkThreshold()
#    thresholdx.SetInputData(normalsch.GetOutput())
#    #thresholdx.SetAttributeModeToUseCellData()
#    #thresholdx.ThresholdBetween(-0.95,0.95)
#    thresholdx.ThresholdBetween(0.95,1)
#    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
#    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "zn")
#    thresholdx.Update()
#
#
#    esx2 = vtk.vtkDataSetSurfaceFilter()
#    esx2.SetInputConnection(thresholdx.GetOutputPort())
#    esx2.Update()
#  
#    LogVTK( esx2.GetOutput(), directoryForLog+"convexhullfilter.vtp")
##    writer_td = vtk.vtkXMLPolyDataWriter()
##    #writer = vtk.vtkSimplePointsWriter()
##    writer_td.SetInputData(esx2.GetOutput())
##    writer_td.SetFileName(directoryForLog+"convexhullfilter.vtp")
##    writer_td.Write()
##    
#    return(esx2.GetOutput())
    
     
    thresholdy = vtk.vtkThreshold()
    thresholdy.SetInputData(normalsch.GetOutput())
   
    thresholdy.ThresholdBetween(-0.4,0.4)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.Update()
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(thresholdy.GetOutput())
   
    #thresholdx.ThresholdBetween(-0.4,0.4)
    thresholdx.ThresholdBetween(-1.0,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
#    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdx.GetOutput())
    #thresholdx.SetAttributeModeToUseCellData()
    #thresholdx.ThresholdBetween(-0.95,0.95)
    thresholdz.ThresholdBetween(-1,0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    
    


    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdz.GetOutputPort())
    esz2.Update()
  
    LogVTK( esz2.GetOutput(), directoryForLog+"convexhullfilter.vtp")
#    writer_td = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    writer_td.SetInputData(esx2.GetOutput())
#    writer_td.SetFileName(directoryForLog+"convexhullfilter.vtp")
#    writer_td.Write()
#    
    return(esz2)
    #return(normalsch)
    



def ClipOrigMeshGrooves(gb, origMeshSurface):
    
    
    
    tireClippedToOnlyGrooves=vtk.vtkAppendPolyData()
    
    for i in range(0,len(gb)):
        print "\n in ClipOrg ", i
        bounds=gb[i].GetOutput().GetBounds()
        print "\nbounds are ", bounds
        box=vtk.vtkBox()
        box.SetBounds(bounds)
        
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(origMeshSurface.GetOutputPort())
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(1)
        clipper.Update()
        
        es1 = vtk.vtkDataSetSurfaceFilter()
        es1.SetInputConnection(clipper.GetOutputPort())
        es1.Update()
        
        LogVTK(tireClippedToOnlyGrooves.GetOutput(), directoryForLog+"original_grooves_clipped_"+str(i)+"_.vtp")
        
#        writer_td87 = vtk.vtkXMLPolyDataWriter()
#        writer_td87.SetInputData(tireClippedToOnlyGrooves.GetOutput())
#        filename = directoryForLog+"original_grooves_clipped_"+str(i)+"_.vtp"
#        #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#        writer_td87.SetFileName(filename)
#        writer_td87.Write()
        
        tireClippedToOnlyGrooves.AddInput(es1.GetOutput())
        tireClippedToOnlyGrooves.Update()
    
    LogVTK(tireClippedToOnlyGrooves.GetOutput(), directoryForLog+"original_grooves_clipped_all.vtp")
#    writer_td86 = vtk.vtkXMLPolyDataWriter()
#    writer_td86.SetInputData(tireClippedToOnlyGrooves.GetOutput())
#    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
#    writer_td86.SetFileName(directoryForLog+"original_grooves_clipped_all.vtp")
#    writer_td86.Write()
    
    return(tireClippedToOnlyGrooves)
  

 

def ClipMesh(filename,startx,endx):
    
    
    ## !!!!!  for now , adjust this to account for the fact that older TireAudit mechanism
    # had longfer frame (114mm vs 85mm)
    
    print "filename",filename
    
    tireAuditFrameHeight=0.085
    #tireAuditFrameHeight=0.114
     
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    originalMesh = reader.GetOutput()
        
        
    reader.Update()
    
    print "\n^^^^^^^^# of pts", reader.GetOutput().GetNumberOfPoints(),"\n"
    
    LogVTK(reader.GetOutput(), directoryForLog+"originalafterread.vtp")
    
    # clip within frame
    
    # Generate Surface Normals
    # xyrat threshold 
    # Z Threshold
    
  
    # if 0 use supplied defaults for clipping

    xmin=startx
    xmax=endx

    
    yTargetCorrection=0.022
    ymin = 0.0 + yTargetCorrection
    #ymax = 0.084 - yTargetCorrectionrgetCorrection
    ymax = tireAuditFrameHeight  - yTargetCorrection
         
    zmax=0.005  
    zmin=zmax-0.06
      
     
        
    yThreshold=(ymax-ymin)*0.8
    print "\nyThreshold",yThreshold
        
    box=vtk.vtkBox()  
    box.SetBounds(xmin,xmax,ymin,ymax,zmin,zmax)
    
    print "\nbounds", xmin,xmax,ymin,ymax,zmin,zmax,"\n"
    
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputConnection(reader.GetOutputPort())
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(1)
    clipper.Update()
    
    
    
    es1 = vtk.vtkDataSetSurfaceFilter()
    es1.SetInputConnection(clipper.GetOutputPort())
    es1.Update()
    
    cleanedMesh=KeepOnlyLargestConnectedComponent(clipper)
    
    LogVTK(cleanedMesh.GetOutput(), directoryForLog+"After_clipped.vtp")
    LogVTK(cleanedMesh.GetOutput(), directoryForLog+"After_clippedqrs.vtp")
#    writer_td43 = vtk.vtkXMLPolyDataWriter()
#        #write2 = vtk.vtkSimplePointsWriter()
#    writer_td43.SetInputData(cleanedMesh.GetOutput())
#    writer_td43.SetFileName(directoryForLog+"After_clipped.vtp")
#    writer_td43.Write()

    return(originalMesh, cleanedMesh)


def KeepOnlyLargestConnectedComponent1(pd):
    
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(pd)
    
    connectivityFilter.SetExtractionModeToLargestRegion()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    es1 = vtk.vtkDataSetSurfaceFilter()
    es1.SetInputConnection(connectivityFilter.GetOutputPort())
    es1.Update()   
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(es1.GetOutputPort())
    cf1.Update()
    
    LogVTK(es1.GetOutput(), directoryForLog+"CleanedAndClipped.vtp")
    
#    writer_td73 = vtk.vtkXMLPolyDataWriter()
#        #write2 = vtk.vtkSimplePointsWriter()
#    writer_td73.SetInputData(es1.GetOutput())
#    writer_td73.SetFileName(directoryForLog+"CleanedAndClipped.vtp")
#    writer_td73.Write()
        
    
    return(cf1.GetOutput())

def KeepOnlyLargestConnectedComponent(mesh):
    
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(mesh.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToLargestRegion()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    es1 = vtk.vtkDataSetSurfaceFilter()
    es1.SetInputConnection(connectivityFilter.GetOutputPort())
    es1.Update()   
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(es1.GetOutputPort())
    cf1.Update()
    
    LogVTK(es1.GetOutput(), directoryForLog+"CleanedAndClipped.vtp")
    
#    writer_td73 = vtk.vtkXMLPolyDataWriter()
#        #write2 = vtk.vtkSimplePointsWriter()
#    writer_td73.SetInputData(es1.GetOutput())
#    writer_td73.SetFileName(directoryForLog+"CleanedAndClipped.vtp")
#    writer_td73.Write()
        
    
    return(cf1)
    
    #
def ExtractShoulders(es1):
    
   
    
    numPts =  es1.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToClosestPointRegion()
    
    connectivityFilter.SetClosestPoint(float(1.0),float(0.0),float(0.0) )
    
    cfr = vtk.vtkCleanPolyData()
    cfr.SetInputConnection(connectivityFilter.GetOutputPort())
    cfr.Update()
    
    r=vtk.vtkPolyData()
    r.DeepCopy(cfr.GetOutput())
        
    numPts =  cfr.GetOutput().GetNumberOfPoints()
    print "\n************** num pts after clean", numPts
        
    LogVTK(cfr.GetOutput(),  "c:\\temp\\rightshoulder.vtp")
    
    
    connectivityFilter.SetClosestPoint(float(0.0),float(0.0),float(0.0) )
    
    cfl = vtk.vtkCleanPolyData()
    cfl.SetInputConnection(connectivityFilter.GetOutputPort())
    cfl.Update()
        
    numPts =  cfl.GetOutput().GetNumberOfPoints()
    print "\n************** num pts after clean", numPts
        
    LogVTK(cfl.GetOutput(),  "c:\\temp\\leftshoulder.vtp")
   
           
    
#    
    return(cfl.GetOutput(),r)

def ExtractShoulders1(es1):
    
   
    
    numPts =  es1.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToClosestPointRegion()
    
    connectivityFilter.SetClosestPoint(0.14,0,0)
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(connectivityFilter.GetOutputPort())
    cf1.Update()
        
    numPts =  cf1.GetOutput().GetNumberOfPoints()
    print "\n************** num pts after clean", numPts
        
    LogVTK(connectivityFilter.GetOutput(),  "c:\\temp\\rightshoulder.vtp")
           
    
#    
    return()


def ReturnNumberOfConnectedComponents(pd):
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(pd)
    
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.Update()
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of cc\n ", int(ner)
    return(ner)
    
def ReturnNumberOfConnectedComponentsAndSizes(pd):
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(pd)
    
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.Update()
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of cc\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
             
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
    
    
    
    return((ner,region_size))

        

def ExtractLargestNComponents(es1, numComponents,sizeThreshold):
    
   
    
    numPts =  es1.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    
    
    connectivityFilter1 = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter1.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter1.SetExtractionModeToSpecifiedRegions()
    connectivityFilter1.ColorRegionsOn()
    
    connectivityFilter1.ColorRegionsOn()
    connectivityFilter1.Update()
    
  
   
        
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
    
    
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)
    #sorted_by_cf_regionsize = sorted(region_size, key=lambda tup: tup[1]) 
    #def getKey(item):
    #    return (item[0],item[1])
    #sorted_by_cf_regionsize=sorted(region_size, key=lambda x: x[1])
    
    print "\n** sorted by second **\n"
    
    numregtoisolate=ner
    numComponents,sizeThreshold
    if (numComponents>0):
        numregtoisolate=numComponents
    else:
        for cnt,item in enumerate(sorted_by_cf_regionsize):
            if item[1]<sizeThreshold:
                numregtoisolate=cnt
                break
    
    
    
    #print  sorted_by_cf_regionsize[int(ner)-1]
    #print  sorted_by_cf_regionsize[int(ner)-2]
    for j in range(0,numregtoisolate):
        print "******************", sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    
    connectivityFilter.Update()
    
    
       
   
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions after Init\n ", int(ner)
    
    
    
    
    print "\n number of regions\n ", int(ner)
    
    

    
    for j in range(0,numregtoisolate):
    #for j in range(0,numComponents):
        
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]
        
       
    
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(es1.GetOutputPort())
        cf1.Update()
        
    
        
        bbOneRegion=[0,0,0,0,0,0]
        bbOneRegion=connectivityFilter.GetOutput().GetBounds()
        print "\n bounds of region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
        numPts =  connectivityFilter.GetOutput().GetNumberOfPoints()
        print "\nnum pts before clean", numPts
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(connectivityFilter.GetOutputPort())
        cf1.Update()
        
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum pts after clean", numPts
        
      
        
        
        
        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        connectivityFilter1.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        
        LogVTK(connectivityFilter.GetOutput(),  directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp")
        

        
        
        if (j<>0):
             connectivityFilter1.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
             connectivityFilter1.Update()
             cf1 = vtk.vtkCleanPolyData()
             cf1.SetInputConnection(connectivityFilter1.GetOutputPort())
             cf1.Update()
             numPts =  cf1.GetOutput().GetNumberOfPoints()
             print "\nnum in current region", numPts
     
            
             
        LogVTK(cf1.GetOutput(), directoryForLog+"regionDebug_"+str(j)+".vtp")
#    

        
    
    connectivityFilter.Update()
    
    
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(connectivityFilter.GetOutputPort())
    cf1.Update()
    
    LogVTK(cf1.GetOutput(), "c:\\temp\\regionsaggregated.vtp")
    LogVTK(cf1.GetOutput(), directoryForLog+ "regionsaggregated.vtp")
#    writer_td31 = vtk.vtkXMLPolyDataWriter()
#        #writer = vtk.vtkSimplePointsWriter()
#    writer_td31.SetInputData(cf1.GetOutput())
#    writer_td31.SetFileName(directoryForLog+"regionsaggregated.vtp")
#    writer_td31.Write()
#    
    return(cf1)
    
    
def RemoveTinyComponents(es1, sizeThreshold):
    
   
    
    numPts =  es1.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    
    
    connectivityFilter1 = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter1.SetInputConnection(es1.GetOutputPort() )
    
    connectivityFilter1.SetExtractionModeToSpecifiedRegions()
    connectivityFilter1.ColorRegionsOn()
    
    connectivityFilter1.ColorRegionsOn()
    connectivityFilter1.Update()
    
  
   
        
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
    
    
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)
    #sorted_by_cf_regionsize = sorted(region_size, key=lambda tup: tup[1]) 
    #def getKey(item):
    #    return (item[0],item[1])
    #sorted_by_cf_regionsize=sorted(region_size, key=lambda x: x[1])
    
    print "\n** sorted by second **\n"
    
    numregtoisolate=ner
    #print  sorted_by_cf_regionsize[int(ner)-1]
    #print  sorted_by_cf_regionsize[int(ner)-2]
    for j in range(0,numregtoisolate):
        print  sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    
    connectivityFilter.Update()
    
    
       
   
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions after Init\n ", int(ner)
    
    
    
    
    print "\n number of regions\n ", int(ner)
    
    

    
    for j in range(0,numregtoisolate):
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]
        
        if (sorted_by_cf_regionsize[j][1]<sizeThreshold):
            break
        
       
    
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(es1.GetOutputPort())
        cf1.Update()
        
    
        
        bbOneRegion=[0,0,0,0,0,0]
        bbOneRegion=connectivityFilter.GetOutput().GetBounds()
        print "\n bounds of region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
        numPts =  connectivityFilter.GetOutput().GetNumberOfPoints()
        print "\nnum pts before clean", numPts
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(connectivityFilter.GetOutputPort())
        cf1.Update()
        
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum pts after clean", numPts
        
        
        
        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        connectivityFilter1.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        
        LogVTK(connectivityFilter.GetOutput(),  directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp")
        

        
        
        if (j<>0):
             connectivityFilter1.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
             connectivityFilter1.Update()
             cf1 = vtk.vtkCleanPolyData()
             cf1.SetInputConnection(connectivityFilter1.GetOutputPort())
             cf1.Update()
             numPts =  cf1.GetOutput().GetNumberOfPoints()
             print "\nnum in current region", numPts
     
            
             
        LogVTK(cf1.GetOutput(), directoryForLog+"regionDebug_"+str(j)+".vtp")
#    

        
    
    connectivityFilter.Update()
    
    
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(connectivityFilter.GetOutputPort())
    cf1.Update()
    
    LogVTK(cf1.GetOutput(), "c:\\temp\\regionsaggregated.vtp")
    LogVTK(cf1.GetOutput(), directoryForLog+ "regionsaggregated.vtp")
#    writer_td31 = vtk.vtkXMLPolyDataWriter()
#        #writer = vtk.vtkSimplePointsWriter()
#    writer_td31.SetInputData(cf1.GetOutput())
#    writer_td31.SetFileName(directoryForLog+"regionsaggregated.vtp")
#    writer_td31.Write()
#    
    return(cf1)
    

def ExtractLargestNComponents1(pd, numComponents,sizeThreshold):
    
   
    
    numPts =  pd.GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(pd)
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    
    
    connectivityFilter1 = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter1.SetInputData(pd)
    
    connectivityFilter1.SetExtractionModeToSpecifiedRegions()
    connectivityFilter1.ColorRegionsOn()
    
    connectivityFilter1.ColorRegionsOn()
    connectivityFilter1.Update()
    
   
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
    
    
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)
    #sorted_by_cf_regionsize = sorted(region_size, key=lambda tup: tup[1]) 
    #def getKey(item):
    #    return (item[0],item[1])
    #sorted_by_cf_regionsize=sorted(region_size, key=lambda x: x[1])
    
    print "\n** sorted by second **\n"
    
    #numregtoisolate=min(ner,numComponents)
   
    if (numComponents>0):
        numregtoisolate=numComponents
    else:
        numregtoisolate=ner
        for cnt,item in enumerate(sorted_by_cf_regionsize):
            if item[1]<sizeThreshold:
                numregtoisolate=cnt
                break
    
    
    
    #print  sorted_by_cf_regionsize[int(ner)-1]
    #print  sorted_by_cf_regionsize[int(ner)-2]
#    for j in range(numregtoisolate):
#        print "******************", sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    
    connectivityFilter.Update()
    
    
       
   
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions after Init\n ", int(ner)
    
    
    
    
    print "\n number of regions\n ", int(ner)
    
    

    pdList=[]
    for j in range(0,numregtoisolate):
    #for j in range(0,numComponents):
        
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]
        
       
    
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputData(pd)
        cf1.Update()
        
    
        
        bbOneRegion=[0,0,0,0,0,0]
        bbOneRegion=connectivityFilter.GetOutput().GetBounds()
        print "\n bounds of region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
        numPts =  connectivityFilter.GetOutput().GetNumberOfPoints()
        print "\nnum pts before clean", numPts
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(connectivityFilter.GetOutputPort())
        cf1.Update()
        
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum pts after clean", numPts
        
        
        
        
        
        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        connectivityFilter1.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        connectivityFilter1.Update()
        
        LogVTK(connectivityFilter1.GetOutput(),  directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp")
        

        
#        
#        if (j<>0):
#             connectivityFilter1.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
#             connectivityFilter1.Update()
#             cf1 = vtk.vtkCleanPolyData()
#             cf1.SetInputConnection(connectivityFilter1.GetOutputPort())
#             cf1.Update()
#             numPts =  cf1.GetOutput().GetNumberOfPoints()
#             print "\nnum in current region", numPts
#             
#             pdList.append(cf1.GetOutput())
#        else:
#             pdList.append(connectivityFilter.GetOutput())
        if (j<>0):
             connectivityFilter1.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
             connectivityFilter1.Update()
        cf1 = vtk.vtkCleanPolyData() 
        cf1.SetInputConnection(connectivityFilter1.GetOutputPort())
        cf1.Update()
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum in current region", numPts
             
        pdList.append(cf1.GetOutput())
      
            
             
        LogVTK(cf1.GetOutput(), directoryForLog+"regionDebug_"+str(j)+".vtp")
#    

        
    
    connectivityFilter.Update()
    
    
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(connectivityFilter.GetOutputPort())
    cf1.Update()
    
    LogVTK(cf1.GetOutput(), "c:\\temp\\regionsaggregated.vtp")
    LogVTK(cf1.GetOutput(), directoryForLog+ "regionsaggregated.vtp")
#    writer_td31 = vtk.vtkXMLPolyDataWriter()
#        #writer = vtk.vtkSimplePointsWriter()
#    writer_td31.SetInputData(cf1.GetOutput())
#    writer_td31.SetFileName(directoryForLog+"regionsaggregated.vtp")
#    writer_td31.Write()
#    
    #return(cf1)
    return(pdList)

def ExtractLargestNComponents2(pd, numComponents,sizeThreshold):
    
   
    
    numPts =  pd.GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(pd)
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    
    
    connectivityFilter1 = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter1.SetInputData(pd)
    
    connectivityFilter1.SetExtractionModeToSpecifiedRegions()
    connectivityFilter1.ColorRegionsOn()
    
    connectivityFilter1.ColorRegionsOn()
    connectivityFilter1.Update()
    
   
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
    
    
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)

    print "\n** sorted by second **\n"
    
    numregtoisolate=ner
    numComponents,sizeThreshold
    if (numComponents>0):
        numregtoisolate=numComponents
    else:
        for cnt,item in enumerate(sorted_by_cf_regionsize):
            if item[1]<sizeThreshold:
                numregtoisolate=cnt
                break
    
    
    for j in range(0,numregtoisolate):
        print "******************", sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    
    connectivityFilter.Update()
    
    
       
    connectivityFilter1.InitializeSpecifiedRegionList()
    connectivityFilter1.Update()
    
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions after Init\n ", int(ner)

    
    print "\n number of regions\n ", int(ner)
      

    pdList=[]
    for j in range(0,numregtoisolate):
    #for j in range(0,numComponents):
        
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]
        
       

        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        #connectivityFilter1.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        
        LogVTK(connectivityFilter.GetOutput(),  directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp")
        


        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(connectivityFilter.GetOutputPort())
        cf1.Update()
        pdList.append(cf1.GetOutput())
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum in current region", numPts
        bbox=cf1.GetOutput().GetBounds()
        print ("\n boundinb box for region ", j, "is ",bbox)
        connectivityFilter.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
        connectivityFilter.Update()
            
             
        LogVTK(cf1.GetOutput(), directoryForLog+"regionDebug_"+str(j)+".vtp")
#    

        
    
    connectivityFilter.Update()
    
    
    
    cfLargestNRegions = vtk.vtkCleanPolyData()
    cfLargestNRegions.SetInputConnection(connectivityFilter1.GetOutputPort())
    cfLargestNRegions.Update()
    
    LogVTK(cfLargestNRegions.GetOutput(), "c:\\temp\\regionsaggregated.vtp")
    LogVTK(cfLargestNRegions.GetOutput(), directoryForLog+ "regionsaggregated.vtp")
#    writer_td31 = vtk.vtkXMLPolyDataWriter()
#        #writer = vtk.vtkSimplePointsWriter()
#    writer_td31.SetInputData(cf1.GetOutput())
#    writer_td31.SetFileName(directoryForLog+"regionsaggregated.vtp")
#    writer_td31.Write()
#    
    return(cfLargestNRegions.GetOutput())
    #return(pdList)  


def RemoveGroovesFromMesh(es1):
    
   

    tireNormals = vtk.vtkPolyDataNormals()
    
    tireNormals.SetInputConnection(es1.GetOutputPort())
    tireNormals.ComputeCellNormalsOn()
    tireNormals.SetFeatureAngle (23)
    tireNormals.SplittingOn ()
    tireNormals.Update()
    
    
    
    normCells = tireNormals.GetOutput().GetCellData().GetNormals()
    areaCells=tireNormals.GetOutput().GetCellData().GetArray(0)
    
    numCells =  tireNormals.GetOutput().GetNumberOfCells()
    data = tireNormals.GetOutput()
    
    print "\n", "number of cells", numCells
    # convert normal vectors to 3 scalar arrays
    
    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")
    
    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")
    
    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")
    
    
    itemList=[]
    for i in range (0,numCells):
    
        a0 = areaCells.GetTuple(i)
        area=a0[0]
        ##print "\narea", area,"\n"
        n0=normCells.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"
    
        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)
    
        itemEntry=(i,x,y,z,area)
        itemList.append(itemEntry)
    
    
    itemList.sort(key=lambda tup: tup[4])
    
    
    tireNormals.GetOutput().GetCellData().AddArray(data1)
    tireNormals.GetOutput().GetCellData().AddArray(data2)
    tireNormals.GetOutput().GetCellData().AddArray(data3)
    
    
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(tireNormals.GetOutput())
    thresholdx.ThresholdBetween(-0.5,0.5)
    

    
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "xn")
    thresholdx.Update()
    
    es_thresholdx = vtk.vtkDataSetSurfaceFilter()
    es_thresholdx.SetInputConnection(thresholdx.GetOutputPort())
    es_thresholdx.Update()
    
    LogVTK(es_thresholdx.GetOutput(),directoryForLog+"threshtest_after_x.vtp")
    
#    writer_td33 = vtk.vtkXMLPolyDataWriter()
#        #write2 = vtk.vtkSimplePointsWriter()
#    writer_td33.SetInputData(es_thresholdx.GetOutput())
#    writer_td33.SetFileName(directoryForLog+"threshtest_after_x.vtp")
#    writer_td33.Write()
#    
    
    
    thresholdz = vtk.vtkThreshold()
    thresholdz.SetInputData(thresholdx.GetOutput())
    thresholdz.ThresholdBetween(0.9,1.0)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdz.SetInputArrayToProcess(0, 0, 0, 1, "zn")
    thresholdz.Update()
    
    
    
    es_thresholdz = vtk.vtkDataSetSurfaceFilter()
    es_thresholdz.SetInputConnection(thresholdz.GetOutputPort())
    es_thresholdz.Update()
    
    LogVTK(es_thresholdz.GetOutput(),directoryForLog+"threshtest_after_xz.vtp")
    
#    writer_td34 = vtk.vtkXMLPolyDataWriter()
#        #write2 = vtk.vtkSimplePointsWriter()
#    writer_td34.SetInputData(es_thresholdz.GetOutput())
#    writer_td34.SetFileName(directoryForLog+"threshtest_after_xz.vtp")
#    writer_td34.Write()
    
    
    
    thresholdy = vtk.vtkThreshold()
    thresholdy.SetInputData(thresholdx.GetOutput())
    thresholdy.ThresholdBetween(-0.2,0.2)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdy.Update()
    
    
    
    es_thresholdy = vtk.vtkDataSetSurfaceFilter()
    es_thresholdy.SetInputConnection(thresholdy.GetOutputPort())
    es_thresholdy.Update()
    
    LogVTK(es_thresholdy.GetOutput(),directoryForLog+"threshtest_after_xzy.vtp")
    
#    writer_td35 = vtk.vtkXMLPolyDataWriter()
#        #write5 = vtk.vtkSimplePointsWriter()
#    writer_td35.SetInputData(es_thresholdy.GetOutput())
#    writer_td35.SetFileName(directoryForLog+"threshtest_after_xzy.vtp")
#    writer_td35.Write()
#    
#    
#    
# 
    
     
    #numberOfPointArrays=reader.GetOutput().GetPointData().GetNumberOfArrays()
    #print "\n number of Point Arrays\n", numberOfPointArrays
    #
    #    
    #distPts = reader.GetOutput().GetPointData().GetArray(1)
    
    numPts =  es_thresholdy.GetOutput().GetNumberOfPoints()  
    print "\n number of points", numPts  
    
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(es_thresholdy.GetOutputPort() )
    
    connectivityFilter.SetExtractionModeToSpecifiedRegions()
    connectivityFilter.ColorRegionsOn()
    
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()
    
    
    
    connectivityFilter1 = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter1.SetInputConnection(es_thresholdy.GetOutputPort() )
    
    connectivityFilter1.SetExtractionModeToSpecifiedRegions()
    connectivityFilter1.ColorRegionsOn()
    
    connectivityFilter1.ColorRegionsOn()
    connectivityFilter1.Update()
    
  
    
    #
    #connectivityFilterOneRegion = vtk.vtkPolyDataConnectivityFilter()
    #connectivityFilterOneRegion.SetInputConnection(es2.GetOutputPort() )
    #connectivityFilterOneRegion.SetExtractionModeToSpecifiedRegions()
    #connectivityFilterOneRegion.ColorRegionsOn()
    #connectivityFilterOneRegion.Update()
    #    
        
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions\n ", int(ner)
    
    fa=vtk.vtkFloatArray()
    fa.SetNumberOfComponents(1);
    fa.SetNumberOfValues(int(ner))
    for j in range(ner):
            fa.SetValue(j, float(j))
            
        
    
    
        
    fa=connectivityFilter.GetRegionSizes ()
    
    region_size=[ (0,0) for i in range(int(ner))]
    for j in range(0,int(ner)):
            #print j,fa.GetValue(j)
            region_size[j]=(j,fa.GetValue(j))
            
    def getKey(item):
        return item[1]
    sorted_by_cf_regionsize=sorted(region_size, key=getKey,reverse=True)
    #sorted_by_cf_regionsize = sorted(region_size, key=lambda tup: tup[1]) 
    #def getKey(item):
    #    return (item[0],item[1])
    #sorted_by_cf_regionsize=sorted(region_size, key=lambda x: x[1])
    
    print "\n** sorted by second **\n"
    
    numregtoisolate=30
    #print  sorted_by_cf_regionsize[int(ner)-1]
    #print  sorted_by_cf_regionsize[int(ner)-2]
    for j in range(0,numregtoisolate):
        print  sorted_by_cf_regionsize[j]
    
    
    connectivityFilter.InitializeSpecifiedRegionList()
    
    
    connectivityFilter.Update()
    
    
       
   
    
    
    ner= connectivityFilter.GetNumberOfExtractedRegions ()
    print "\n number of regions after Init\n ", int(ner)
    
    
    
    
    print "\n number of regions\n ", int(ner)
    
    
    widthThresh=0.006
    xyRatioThresh=0.33
    #xyRatioThresh=0.25
    saBBRatioRatioThresh=0.25
    saThresh=0.00002
    
    for j in range(0,numregtoisolate):
        print "\nOrdinal Region Number Sort Order", j
        print  "\nRegionID", sorted_by_cf_regionsize[j][0]
        #sorted_by_cf_regionsize[i][0]
        
       
    
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(es_thresholdy.GetOutputPort())
        cf1.Update()
        
    
        
        bbOneRegion=[0,0,0,0,0,0]
        bbOneRegion=connectivityFilter.GetOutput().GetBounds()
        print "\n bounds of region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
        numPts =  connectivityFilter.GetOutput().GetNumberOfPoints()
        print "\nnum pts before clean", numPts
        
        cf1 = vtk.vtkCleanPolyData()
        cf1.SetInputConnection(connectivityFilter.GetOutputPort())
        cf1.Update()
        
        numPts =  cf1.GetOutput().GetNumberOfPoints()
        print "\nnum pts after clean", numPts
        
        
        
        
        connectivityFilter.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        connectivityFilter1.AddSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        
        LogVTK(connectivityFilter.GetOutput(),  directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp")
        
#        writer_td45 = vtk.vtkXMLPolyDataWriter()
#        #write5 = vtk.vtkSimplePointsWriter()
#        writer_td45.SetInputData(connectivityFilter.GetOutput())
#        filename=directoryForLog+"threshtest_afterconnecityfilter_"+str(j)+"_.vtp"
#        writer_td45.SetFileName(filename)
#        writer_td45.Write()
        
            
        
        
        if (j<>0):
             connectivityFilter1.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j-1][0]))
             connectivityFilter1.Update()
             cf1 = vtk.vtkCleanPolyData()
             cf1.SetInputConnection(connectivityFilter1.GetOutputPort())
             cf1.Update()
             numPts =  cf1.GetOutput().GetNumberOfPoints()
             print "\nnum in current region", numPts
             bbOneRegion=[0,0,0,0,0,0]
             bbOneRegion=cf1.GetOutput().GetBounds()
             print "\n bounds of current region\n", int(sorted_by_cf_regionsize[j][0]), bbOneRegion
             print "\n bounds of current region\n", abs(bbOneRegion[0]-bbOneRegion[1]), abs(bbOneRegion[2]-bbOneRegion[3]),abs(bbOneRegion[4]-bbOneRegion[5])
             xyRatio=abs(bbOneRegion[0]-bbOneRegion[1])/abs(bbOneRegion[2]-bbOneRegion[3])
             massProperty = vtk.vtkMassProperties()
             massProperty.SetInputConnection(cf1.GetOutputPort())
             massProperty.Update()
             sa = massProperty.GetSurfaceArea()
             print "\nSurface Area", sa
             saBBRatio = sa/ ( (bbOneRegion[1]-bbOneRegion[0]) * (bbOneRegion[3]-bbOneRegion[2])  )
             print "\nBBArea", (bbOneRegion[1]-bbOneRegion[0]) * (bbOneRegion[3]-bbOneRegion[2]) 
             print "\nsaBBRatiio", saBBRatio
             
             LogVTK(cf1.GetOutput(), directoryForLog+"regionDebug_"+str(j)+".vtp")
#    
#             writer_td51 = vtk.vtkXMLPolyDataWriter()
#             #writer = vtk.vtkSimplePointsWriter()
#             filename=directoryForLog+"regionDebug_"+str(j)+".vtp"
#             writer_td51.SetInputData(cf1.GetOutput())
#             writer_td51.SetFileName(filename)
#             writer_td51.Write()
#    
             
             
             print "\n xy Ratios", xyRatio,1/xyRatio, xyRatioThresh,1/xyRatioThresh
             print "\n sa Ratios", saBBRatio, saBBRatioRatioThresh, sa, saThresh
             #if ( (xyRatio<xyRatioThresh) | ((1/xyRatio)>(1/xyRatioThresh) ) |
             if (  ((1/xyRatio)>(1/xyRatioThresh) ) |  (  (bbOneRegion[1]-bbOneRegion[0]) < widthThresh  )  |
                 (saBBRatio<saBBRatioRatioThresh)| (sa<saThresh) ):
                 print "\nRemoving region", int(sorted_by_cf_regionsize[j][0])
             
                 # remove narrow or thin parts  OR filamenty parts 
                 connectivityFilter.DeleteSpecifiedRegion(int(sorted_by_cf_regionsize[j][0]))
        
    
    connectivityFilter.Update()
    
    
    
    cf1 = vtk.vtkCleanPolyData()
    cf1.SetInputConnection(connectivityFilter.GetOutputPort())
    cf1.Update()
    
    LogVTK(cf1.GetOutput(), "c:\\temp\\regionsaggregated.vtp")
    LogVTK(cf1.GetOutput(), directoryForLog+ "regionsaggregated.vtp")
#    writer_td31 = vtk.vtkXMLPolyDataWriter()
#        #writer = vtk.vtkSimplePointsWriter()
#    writer_td31.SetInputData(cf1.GetOutput())
#    writer_td31.SetFileName(directoryForLog+"regionsaggregated.vtp")
#    writer_td31.Write()
#    
    return(cf1)
    

def PCAPD(pd):


    
    numpts = pd.GetNumberOfPoints()
    
    
    m0Name = "M0"
    dataset1Arr = vtk.vtkDoubleArray()
    dataset1Arr.SetNumberOfComponents(1)
    dataset1Arr.SetName( m0Name )
    
    m1Name = "M1"
    dataset2Arr = vtk.vtkDoubleArray()
    dataset2Arr.SetNumberOfComponents(1)
    dataset2Arr.SetName( m1Name )
    
    m2Name = "M2"
    dataset3Arr = vtk.vtkDoubleArray()
    dataset3Arr.SetNumberOfComponents(1)
    dataset3Arr.SetName( m2Name )
    
    print "\n numpts ",numpts
    
    totalX=0.0
    totalY=0.0
    totalZ=0.0
    p=[0.0,0.0,0.0]
    for i in range(0,numpts):
        p = pd.GetPoint(i)
        x,y,z=p[:3]
        totalX=totalX+x
        totalY=totalY+y
        totalZ=totalZ+z
        dataset1Arr.InsertNextValue(x)
        dataset2Arr.InsertNextValue(y)
        dataset3Arr.InsertNextValue(z)
     
    meanX=totalX/(numpts) 
    meanY=totalY/(numpts) 
    meanZ=totalZ/(numpts) 
      
    datasetTable = vtk.vtkTable()
    datasetTable.AddColumn(dataset1Arr)
    datasetTable.AddColumn(dataset2Arr)
    datasetTable.AddColumn(dataset3Arr)
    
    print "\nMeanX",meanX
    print "\nMeanY",meanY
    print "\nMeanZ",meanZ
    
    
    #
    print "\n got here A\n"
    pcaStatistics = vtk.vtkPCAStatistics()
    q_INPUT_DATA=0
    pcaStatistics.SetInputData( q_INPUT_DATA, datasetTable )
    
    print "\n got here B\n"
    # 
    pcaStatistics.SetColumnStatus("M0", 1 )
    pcaStatistics.SetColumnStatus("M1", 1 )
    pcaStatistics.SetColumnStatus("M2", 1 )
    pcaStatistics.RequestSelectedColumns()
    pcaStatistics.SetDeriveOption(True)
    pcaStatistics.Update()
    # 
    eigenvalues = vtk.vtkDoubleArray()
    pcaStatistics.GetEigenvalues(eigenvalues);
    #//  double eigenvaluesGroundTruth[3] = {.5, .166667, 0};
    for i in range(eigenvalues.GetNumberOfTuples() ):
        print "\n", eigenvalues.GetValue(i)
    
     
    #  ///////// Eigenvectors ////////////
    
    eigenvectors=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    eigenvectors = vtk.vtkDoubleArray()
    
    
    evec=[0,0,0]
    
    eigenXYZ=[ ]
    
    pcaStatistics.GetEigenvectors(eigenvectors)
    for i in range(eigenvectors.GetNumberOfTuples() ):
        print "\n Eigenvector",i,":"
        #evec = eigenvectors.GetNumberOfComponents()
        print "\n", eigenvalues.GetValue(i)
        eigenvectors.GetTuple(i,evec)
        for j in range(eigenvectors.GetNumberOfComponents() ):
            print "\n ",evec[j],":"
        eigenXYZvec= ( evec[0],evec[1],evec[2]  )   
        eigenXYZ.append( eigenXYZvec )
       
            
            
            ##eigenvectorSingle=vtk.vtkDoubleArray()
            ##pcaStatistics.GetEigenvector(i,eigenvectorSingle)
        
        print "\n"
    
    print "\nEigen XYZ",eigenXYZ
    
    m= vtk.vtkMatrix4x4()
    
    m.DeepCopy((eigenXYZ[0][0],eigenXYZ[0][1],eigenXYZ[0][2],0.0,
    eigenXYZ[1][0],eigenXYZ[1][1],eigenXYZ[1][2],0.0,
    eigenXYZ[2][0],eigenXYZ[2][1],eigenXYZ[2][2],0.0,            
    0.0,0.0,0.0,1.0))
    
    
    
    
    transform1 =vtk.vtkTransform()
    

    transform1.Translate(-meanX,-meanY,-meanZ)
    transform1.PostMultiply()
    
  
    
    transform1.Concatenate(m)
    
    return(transform1)


def PcaGrooves(boxPD):
    
    
    
#    print "\n$$$$$$$$$$$$$$$$$$$$$ meshbfore", meshBeforeClipBeforeThresh.GetOutput().GetNumberOfPoints()
#    numpts1 = origMeshPD.GetOutput().GetNumberOfPoints()
#    print "\n orig mesh # pts", numpts1

    
    numpts = boxPD.GetOutput().GetNumberOfPoints()
    
    
    m0Name = "M0"
    dataset1Arr = vtk.vtkDoubleArray()
    dataset1Arr.SetNumberOfComponents(1)
    dataset1Arr.SetName( m0Name )
    
    m1Name = "M1"
    dataset2Arr = vtk.vtkDoubleArray()
    dataset2Arr.SetNumberOfComponents(1)
    dataset2Arr.SetName( m1Name )
    
    m2Name = "M2"
    dataset3Arr = vtk.vtkDoubleArray()
    dataset3Arr.SetNumberOfComponents(1)
    dataset3Arr.SetName( m2Name )
    
    print "\n numpts ",numpts
    
    totalX=0.0
    totalY=0.0
    totalZ=0.0
    p=[0.0,0.0,0.0]
    for i in range(0,numpts):
        p = boxPD.GetOutput().GetPoint(i)
        x,y,z=p[:3]
        totalX=totalX+x
        totalY=totalY+y
        totalZ=totalZ+z
        dataset1Arr.InsertNextValue(x)
        dataset2Arr.InsertNextValue(y)
        dataset3Arr.InsertNextValue(z)
     
    meanX=totalX/(numpts) 
    meanY=totalY/(numpts) 
    meanZ=totalZ/(numpts) 
      
    datasetTable = vtk.vtkTable()
    datasetTable.AddColumn(dataset1Arr)
    datasetTable.AddColumn(dataset2Arr)
    datasetTable.AddColumn(dataset3Arr)
    
    print "\nMeanX",meanX
    print "\nMeanY",meanY
    print "\nMeanZ",meanZ
    
    
    #
    print "\n got here A\n"
    pcaStatistics = vtk.vtkPCAStatistics()
    q_INPUT_DATA=0
    pcaStatistics.SetInputData( q_INPUT_DATA, datasetTable )
    
    print "\n got here B\n"
    # 
    pcaStatistics.SetColumnStatus("M0", 1 )
    pcaStatistics.SetColumnStatus("M1", 1 )
    pcaStatistics.SetColumnStatus("M2", 1 )
    pcaStatistics.RequestSelectedColumns()
    pcaStatistics.SetDeriveOption(True)
    pcaStatistics.Update()
    # 
    eigenvalues = vtk.vtkDoubleArray()
    pcaStatistics.GetEigenvalues(eigenvalues);
    #//  double eigenvaluesGroundTruth[3] = {.5, .166667, 0};
    for i in range(eigenvalues.GetNumberOfTuples() ):
        print "\n", eigenvalues.GetValue(i)
    
     
    #  ///////// Eigenvectors ////////////
    
    eigenvectors=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    eigenvectors = vtk.vtkDoubleArray()
    
    
    evec=[0,0,0]
    
    eigenXYZ=[ ]
    
    pcaStatistics.GetEigenvectors(eigenvectors)
    for i in range(eigenvectors.GetNumberOfTuples() ):
        print "\n Eigenvector",i,":"
        #evec = eigenvectors.GetNumberOfComponents()
        print "\n", eigenvalues.GetValue(i)
        eigenvectors.GetTuple(i,evec)
        for j in range(eigenvectors.GetNumberOfComponents() ):
            print "\n ",evec[j],":"
        eigenXYZvec= ( evec[0],evec[1],evec[2]  )   
        eigenXYZ.append( eigenXYZvec )
       
            
            
            ##eigenvectorSingle=vtk.vtkDoubleArray()
            ##pcaStatistics.GetEigenvector(i,eigenvectorSingle)
        
        print "\n"
    
    print "\nEigen XYZ",eigenXYZ
    
    m= vtk.vtkMatrix4x4()
    
    m.DeepCopy((eigenXYZ[0][0],eigenXYZ[0][1],eigenXYZ[0][2],0.0,
    eigenXYZ[1][0],eigenXYZ[1][1],eigenXYZ[1][2],0.0,
    eigenXYZ[2][0],eigenXYZ[2][1],eigenXYZ[2][2],0.0,            
    0.0,0.0,0.0,1.0))
    
    
    
    
    transform1 =vtk.vtkTransform()
    

    transform1.Translate(-meanX,-meanY,-meanZ)
    transform1.PostMultiply()
    
  
    
    transform1.Concatenate(m)
    
    return(transform1)
    
#    transfmesh1 = vtk.vtkTransformPolyDataFilter()
#    #transfmesh.SetInputConnection(reader.GetOutputPort())
#    transfmesh1.SetInputConnection(meshPD.GetOutputPort())
#    
#    transfmesh1.SetTransform(transform1)
#    transfmesh1.Update()
#    es_transfmesh1 = vtk.vtkDataSetSurfaceFilter()
#    es_transfmesh1.SetInputConnection(transfmesh1.GetOutputPort())
#    es_transfmesh1.Update()
#   
#    
#  
##    transfmesh2 = vtk.vtkTransformPolyDataFilter()
##    #transfmesh.SetInputConnection(reader.GetOutputPort())
##    transfmesh2.SetInputConnection(origMeshPD.GetOutputPort())
##    
##    transfmesh2.SetTransform(transform1)
##    transfmesh2.Update()
##    es_transfmesh2 = vtk.vtkDataSetSurfaceFilter()
##    es_transfmesh2.SetInputConnection(transfmesh2.GetOutputPort())
##    es_transfmesh2.Update()
#   
#    
##    transfmesh3 = vtk.vtkTransformPolyDataFilter()
##    #transfmesh.SetInputConnection(reader.GetOutputPort())
##    transfmesh3.SetInputConnection(origMeshPreThresh.GetOutputPort())
##    
##    transfmesh3.SetTransform(transform1)
##    transfmesh3.Update()
##    es_transfmesh3 = vtk.vtkDataSetSurfaceFilter()
##    es_transfmesh3.SetInputConnection(transfmesh3.GetOutputPort())
##    es_transfmesh3.Update()
#    
#    es_transfmesh3=TransformPD(origMeshPreThresh.GetOutput(),transform1)
#    
#    transfmesh4 = vtk.vtkTransformPolyDataFilter()
#    #transfmesh.SetInputConnection(reader.GetOutputPort())
#    transfmesh4.SetInputConnection(meshBeforeClipBeforeThresh.GetOutputPort())
#    
#    transfmesh4.SetTransform(transform1)
#    transfmesh4.Update()
#    es_transfmesh4 = vtk.vtkDataSetSurfaceFilter()
#    es_transfmesh4.SetInputConnection(transfmesh4.GetOutputPort())
#    es_transfmesh4.Update()
#    
#    transfmesh5 = vtk.vtkTransformPolyDataFilter()
#    #transfmesh.SetInputConnection(reader.GetOutputPort())
#    transfmesh5.SetInputConnection(boxPD.GetOutputPort())
#    
#    transfmesh5.SetTransform(transform1)
#    transfmesh5.Update()
#    es_transfmesh5 = vtk.vtkDataSetSurfaceFilter()
#    es_transfmesh5.SetInputConnection(transfmesh5.GetOutputPort())
#    es_transfmesh5.Update()
#   
#    transfmesh6 = vtk.vtkTransformPolyDataFilter()
#    #transfmesh.SetInputConnection(reader.GetOutputPort())
#    transfmesh6.SetInputConnection(tireClippedToOnlyGrooves.GetOutputPort())
#    
#    transfmesh6.SetTransform(transform1)
#    transfmesh6.Update()
#    es_transfmesh6 = vtk.vtkDataSetSurfaceFilter()
#    es_transfmesh6.SetInputConnection(transfmesh6.GetOutputPort())
#    es_transfmesh6.Update()
#   
#    
#    
#    
#    print "\n got here\n"
#    
#    
#   
#    LogVTK(es_transfmesh1.GetOutput(), directoryForLog+"meshReorientedAfterPCA.vtp")
#    
##     #saved_file="c:\\temp\\m1_worn_clean_randtrans_afterpca.vtp"
##    saved_file=directoryForLog+"meshReorientedAfterPCA.vtp"
##    
##    writer_td16 = vtk.vtkXMLPolyDataWriter()
##    writer_td16.SetInputData(es_transfmesh1.GetOutput())
##    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
##    writer_td16.SetFileName(saved_file)
##    writer_td16.Write()
##    
#   
#    LogVTK(es_transfmesh3.GetOutput(), directoryForLog+"origMeshNoThreshPostPCA.vtp")
#
##    writer_td17 = vtk.vtkXMLPolyDataWriter()
##    writer_td17.SetInputData(es_transfmesh3.GetOutput())
##    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
##    writer_td17.SetFileName(directoryForLog+"origMeshNoThreshPostPCA.vtp")
##    writer_td17.Write()
#    
#    LogVTK(es_transfmesh3.GetOutput(), directoryForLog+"origMeshNoThreshPostPCA.ply")
##    writer_td60 = vtk.vtkPLYWriter()
##    writer_td60.SetArrayName("RGB")
##    writer_td60.SetInputData(es_transfmesh3.GetOutput())
##    writer_td60.SetFileName(directoryForLog+"origMeshNoThreshPostPCA.ply")
##    writer_td60.Write()
#    
#    LogVTK(es_transfmesh4.GetOutput(), directoryForLog+"origMeshNoClipNoThresh1.vtp")
##    
##    writer_td18 = vtk.vtkXMLPolyDataWriter()
##    writer_td18.SetInputData(es_transfmesh4.GetOutput())
##    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
##    writer_td18.SetFileName(directoryForLog+"origMeshNoClipNoThresh1.vtp")
##    writer_td18.Write()
#    
#    LogVTK(es_transfmesh5.GetOutput(), directoryForLog+"allBoxesAfterPCA.vtp")
##    writer_td20 = vtk.vtkXMLPolyDataWriter()
##    writer_td20.SetInputData(es_transfmesh5.GetOutput())
##    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
##    writer_td20.SetFileName(directoryForLog+"allBoxesAfterPCA.vtp")
##    writer_td20.Write()
#    
#    LogVTK(es_transfmesh6.GetOutput(), directoryForLog+"tireClippedToGroovesAfterPCA.vtp")
##    writer_td20 = vtk.vtkXMLPolyDataWriter()
##    writer_td20.SetInputData(es_transfmesh6.GetOutput())
##    #writer_td16.SetFileName("C:\\temp\\m1_worn_clean_transf.vtp")
##    writer_td20.SetFileName(directoryForLog+"tireClippedToGroovesAfterPCA.vtp")
##    writer_td20.Write()
##    
#    LogVTK(es_transfmesh6.GetOutput(), directoryForLog+"tireClippedToGroovesAfterPCA.ply")
##    writer_td59 = vtk.vtkPLYWriter()
##    writer_td59.SetArrayName("RGB")
##    writer_td59.SetInputData(es_transfmesh6.GetOutput())
##    writer_td59.SetFileName(directoryForLog+"tireClippedToGroovesAfterPCA.ply")
##    writer_td59.Write()

#    
#    es_transfmesh6_Rotate180 = vtk.vtkTransformPolyDataFilter()
#    es_transfmesh6_Rotate180.SetInputConnection(es_transfmesh6.GetOutputPort())
#    es_transfmesh6_Rotate180.SetTransform(meshImproveRotate180Transform)
#    es_transfmesh6_Rotate180.Update()
#    
#    es_es_transfmesh6_Rotate180 = vtk.vtkDataSetSurfaceFilter()
#    es_es_transfmesh6_Rotate180.SetInputConnection(es_transfmesh6_Rotate180.GetOutputPort())
#    es_es_transfmesh6_Rotate180.Update()
#    
#    writer_td60 = vtk.vtkPLYWriter()
#    writer_td60.SetArrayName("RGB")
#    writer_td60.SetInputData(es_es_transfmesh6_Rotate180.GetOutput())
#    writer_td60.SetFileName(directoryForLog+"tireClippedToGroovesAfterPCA_Rotate180.ply")
#    writer_td60.Write()
#    
    
    return (es_transfmesh5)


def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

def LocateAndCountVerticalGrooves1(groovesPlus,horizontalOffset,numPtsInSliceThreshold,minGrooveWidth,horizontalSliceInterval ):
    
    print "\n%% Entering locateAndCountVerticalGrooves\n"
    
    # start and end the slicing using the offset
    
    
    grooveStartPos=0
    grooveEndPos=9999
    
 
  
    
    
   # horizontalOffset=0.003
    
    # premise:  you can recognize the start and the end of the groove by either NO points in slice
    # OR by a slice with a few points (in which case you're slicing a treadmarker)
    #numPtsInSliceThreshold=300 
    
#    #filename="c:\\temp\\regionsaggregated1.vtp"
#    meshNoGrooves=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(filename)
#    reader.Update()
    
    bbox=[0,0,0,0,0,0]
    bbox=groovesPlus.GetBounds()
    #numcuts=100
    #minGrooveWidth=0.003
    
    numCuts=int((bbox[1]-bbox[0])/horizontalSliceInterval)
    
    minX=bbox[0]+horizontalOffset
    maxX=bbox[1]-horizontalOffset
    xInterval=(maxX-minX)/numCuts
    
    
    #grooveLeft = [[0 for x in range(5)] for x in range(5)] 
    #grooveRight = [[0 for x in range(5)] for x in range(5)] 
    grooveCount=0
    groovePositions=[]
    inGroove=False
    for i in range(0,numCuts):
        
#  
#        print "\n ### cut # ", i
#        print "\ncut pt x", minX+xInterval*float(i)
                
        planeA=vtk.vtkPlane()
        # change Z baxck to 0
        planeA.SetOrigin(minX+xInterval*float(i),0,0)
        planeA.SetNormal(-1,0,0)
        cutterA=vtk.vtkCutter()
        cutterA.SetSortBy(0)
        cutterA.SetCutFunction(planeA)
        cutterA.SetInputData(groovesPlus)
        cutterA.Update()
        
        LogVTK(cutterA.GetOutput(), directoryForLog+"vslice_"+str(i)+ ".vtp")
        
#        writer_cutter1 = vtk.vtkXMLPolyDataWriter()
#        writer_cutter1.SetInputData(cutterA.GetOutput())
#        fname=directoryForLog+"vslice_"+str(i)+ ".vtp"
#        writer_cutter1.SetFileName(fname)
#        writer_cutter1.Write()
                
        numPts = cutterA.GetOutput().GetNumberOfPoints()
        print "\n numPts", numPts
        print "\ningroove", inGroove
        print "\ngroove count", grooveCount
        print "\ngrooveStartPos", grooveStartPos, grooveEndPos, (grooveEndPos-grooveStartPos)
        
        if (numPts>=numPtsInSliceThreshold):
            if not inGroove:
                inGroove=True
                grooveCount=grooveCount+1
                grooveStartPos=minX+xInterval*float(i)
        else:
            if inGroove:
                inGroove=False
                grooveEndPos=minX+xInterval*float(i-1)
                if ((grooveEndPos-grooveStartPos)>minGrooveWidth):
                    groovePositions.append( (grooveStartPos,grooveEndPos))
                    
                    
    
#    for i,groovePositionsItem in groovePositions:
#                
#    clipper = vtk.vtkClipPolyData()
#    clipper.SetInputConnection(reader.GetOutputPort())
#    clipper.SetClipFunction(box)
#    clipper.SetInsideOut(1)
#    clipper.Update()
#    
#    es1 = vtk.vtkDataSetSurfaceFilter()
#    es1.SetInputConnection(clipper.GetOutputPort())
#    es1.Update()
#                     
    print "\n%%%%%%%% Groove Positions", groovePositions
    return(groovePositions)

def LocateAndCountVerticalGrooves(meshNoGrooves):
    
    print "\n%% Entering locateAndCountVerticalGrooves\n"
    
    # start and end the slicing using the offset
    
    
    grooveStartPos=0
    grooveEndPos=9999
    
    
    horizontalOffset=0.003
    
    # premise:  you can recognize the start and the end of the groove by either NO points in slice
    # OR by a slice with a few points (in which case you're slicing a treadmarker)
    numPtsInSliceThreshold=60 
    
#    #filename="c:\\temp\\regionsaggregated1.vtp"
#    meshNoGrooves=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(filename)
#    reader.Update()
    
    bbox=[0,0,0,0,0,0]
    bbox=meshNoGrooves.GetOutput().GetBounds()
    numcuts=100
    minGrooveWidth=0.003
    
    minX=bbox[0]+horizontalOffset
    maxX=bbox[1]-horizontalOffset
    xInterval=(maxX-minX)/numcuts
    
    
    #grooveLeft = [[0 for x in range(5)] for x in range(5)] 
    #grooveRight = [[0 for x in range(5)] for x in range(5)] 
    grooveCount=0
    groovePositions=[]
    inGroove=False
    for i in range(0,numcuts):
        
#  
#        print "\n ### cut # ", i
#        print "\ncut pt x", minX+xInterval*float(i)
                
        planeA=vtk.vtkPlane()
        # change Z baxck to 0
        planeA.SetOrigin(minX+xInterval*float(i),0,0)
        planeA.SetNormal(-1,0,0)
        cutterA=vtk.vtkCutter()
        cutterA.SetSortBy(0)
        cutterA.SetCutFunction(planeA)
        cutterA.SetInputConnection(meshNoGrooves.GetOutputPort())
        cutterA.Update()
        
        LogVTK(cutterA.GetOutput(), directoryForLog+"vslice_"+str(i)+ ".vtp")
        
#        writer_cutter1 = vtk.vtkXMLPolyDataWriter()
#        writer_cutter1.SetInputData(cutterA.GetOutput())
#        fname=directoryForLog+"vslice_"+str(i)+ ".vtp"
#        writer_cutter1.SetFileName(fname)
#        writer_cutter1.Write()
                
        numPts = cutterA.GetOutput().GetNumberOfPoints()
        print "\n numPts", numPts
        print "\ningroove", inGroove
        print "\ngroove count", grooveCount
        print "\ngrooveStartPos", grooveStartPos, grooveEndPos, (grooveEndPos-grooveStartPos)
        
        if (numPts<=numPtsInSliceThreshold):
            if not inGroove:
                inGroove=True
                grooveCount=grooveCount+1
                grooveStartPos=minX+xInterval*float(i)
        else:
            if inGroove:
                inGroove=False
                grooveEndPos=minX+xInterval*float(i-1)
                if ((grooveEndPos-grooveStartPos)>minGrooveWidth):
                    groovePositions.append( (grooveStartPos,grooveEndPos))
                    
                    
    
#    for i,groovePositionsItem in groovePositions:
#                
#    clipper = vtk.vtkClipPolyData()
#    clipper.SetInputConnection(reader.GetOutputPort())
#    clipper.SetClipFunction(box)
#    clipper.SetInsideOut(1)
#    clipper.Update()
#    
#    es1 = vtk.vtkDataSetSurfaceFilter()
#    es1.SetInputConnection(clipper.GetOutputPort())
#    es1.Update()
#                     
    print "\n%%%%%%%% Groove Positions", groovePositions
    return(groovePositions)
    
def SlicePDAtOrigin(pd,location):
               
    bbox=[0,0,0,0,0,0]
    bbox=pd.GetBounds()
    planeA=vtk.vtkPlane()
    planeA.SetOrigin(0,location,0)

    planeA.SetNormal(0,1,0)

            
        
    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(planeA)
    cutterA.SetInputData(pd)
    cutterA.Update()
    return(cutterA.GetOutput())
    
    
def SliceForGapsToIdentifyGrooves(meshNoGroovesCleaned,gp):
    
    print "\n%% Entering VoteGrooveBoundaries\n"
    
#    #filename="c:\\temp\\regionsaggregated1.vtp"
#    reader=vtk.vtkXMLPolyDataReader()
#    reader.SetFileName(filename)
#    reader.Update()
    
    bbox=[0,0,0,0,0,0]
    bbox=meshNoGroovesCleaned.GetOutput().GetBounds()
    numcuts=30
    
    
    minY=bbox[2]
    maxY=bbox[3]
    yInterval=(maxY-minY)/numcuts
    
    
    
    
    sliceList=[]
    for i in range(0,numcuts):
        
        sliceList.append(i)
        sliceList[i]=[]
#        print "\n ### cut # ", i
#        print "\ncut pt y", minY+yInterval*float(i)
#                
        planeA=vtk.vtkPlane()
        # change Z baxck to 0
        planeA.SetOrigin(0,minY+yInterval*float(i),0)
        planeA.SetNormal(0,-1,0)
        cutterA=vtk.vtkCutter()
        cutterA.SetSortBy(0)
        cutterA.SetCutFunction(planeA)
        cutterA.SetInputConnection(meshNoGroovesCleaned.GetOutputPort())
        cutterA.Update()
        
        ptcoord=[0,0,0]
        numPts = cutterA.GetOutput().GetNumberOfPoints()
        
        LogVTK(cutterA.GetOutput(), directoryForLog+"slice_"+str(i)+ ".vtp")
#        writer_cutter = vtk.vtkXMLPolyDataWriter()
#        writer_cutter.SetInputData(cutterA.GetOutput())
#        fname=directoryForLog+"slice_"+str(i)+ ".vtp"
#        writer_cutter.SetFileName(fname)
#        writer_cutter.Write()
#                    
        
    
        ptsList=[   ]
        for j in range (0,numPts):
            pt = cutterA.GetOutput().GetPoint(j)
            ptsList.append((pt[0],pt[1],pt[2]))
            #print ptsList
        # sort by X
        ptsList.sort(key=lambda tup: tup[0])
        
        print "*********Ptslist*****\n"
        
        for j,item in enumerate(ptsList):
            print "\n",j,item
    
#        notInGroove=True
#        pt = cutterA.GetOutput().GetPoint(0)
#        ptcoord=(pt[0],pt[1],pt[2])
#    
#        grooveCount=0
       
        for k,item in enumerate(gp):
           print "\n*** Grove Positions", k,item,gp[k]
           
       
#        numOfGrooves=len(gp)
#        currentGrooveNumber=0
        
        # use the width of the bounding box that coarsely identify the position of the grooves
        # as a conservative estimate of the true width
#        conservativeGrooveStart=gp[currentGrooveNumber][0]
#        conservativeGrooveEnd=gp[currentGrooveNumber][1]
        #conservativeGrooveTolerance=0.003
        grooveMaxWidth=0.015
        grooveMinWidth=0.002
        for j,pnt in enumerate(ptsList):
            currentX=pnt[0]
            currentY=pnt[1]
            currentZ=pnt[2]
            if (j>0):
                grooveWidth=currentX-prevX
                print "\n pt coord", currentX, currentY, currentZ, prevX, prevY, prevZ, grooveWidth
                if (  (grooveWidth> grooveMinWidth) &  (grooveWidth< grooveMaxWidth) ) :
                    
                #print "\n groove info", currentGrooveNumber,conservativeGrooveStart,conservativeGrooveEnd,conservativeGrooveTolerance
#                if ( (prevX+grooveMaxWidth) & (currentX-conservativeGrooveTolerance<conservativeGrooveEnd) ):
                    
#                    grooveCount=grooveCount+1
#                    print "\ngroove count", grooveCount
                    sliceList[i].append( (prevX,prevY,prevZ, currentX,currentY,currentZ)  )
#                    if (numOfGrooves==grooveCount):
#                        break
                    print "\n&&&& sliceList", sliceList
#                    currentGrooveNumber=currentGrooveNumber+1
#                    conservativeGrooveStart=gp[currentGrooveNumber][0]
#                    conservativeGrooveEnd=gp[currentGrooveNumber][1]
#                    conservativeGrooveTolerance=(conservativeGrooveEnd-conservativeGrooveStart)*conservativeGrooveToleranceFactor
#                                            
            prevX=pnt[0]
            prevY=pnt[1]
            prevZ=pnt[2]
        
    print "\n End of Loop \n"
    for j,item in enumerate(sliceList):
        print "\n",j,item
    
    #print "\n%%%%%%%% Groove list\n", sliceList
    return((sliceList,bbox))  

def ProcessGrooveVotes(glv,groovePositions):
    
  
    grooveXDeviation=0.002
    #processedGrooveList = [[0.0 for numGroovePositions in range(len(groovePositions))] for leftOrRight in range(2)]
    #processedGrooveList = [[[0.0 for numGroovePositions in range(len(groovePositions))] for leftOrRight in range(2)] for numSlices in range(glv)]
    #processedGrooveList = [[[[0.0 for numGroovePositions in range(len(groovePositions))] for leftOrRight in range(2)] for numSlices in range(len(glv))] for numVotesPerSlice in range(maxNumVotesPerSlice) ] 
    #processedGrooveList = ([0 for numGroovePositions in range(len(groovePositions))],[0 for leftOrRight in range(2)],[0 for numSlices in range(len(glv))],[ (0.0,0.0,0.0) for numVotesPerSlice in range(maxNumVotesPerSlice) ] ) 
   #foo = ([0 for j in range(10)], [0 for k in range(100)])
#foo=[[1,2,3],[4,5,6]]
    #processedGrooveList = [[[[[0.0 for numGroovePositions in range(len(groovePositions))] for leftOrRight in range(2)] for numSlices in range(len(glv))] for numVotesPerSlice in range(maxNumVotesPerSlice) ]  for xyz in range(3)]
    #processedGrooveList =  [[0 for numGroovePositions in range(len(groovePositions))] for leftOrRight in range(2)] 
    processedGrooveList =  [[ 0 for leftOrRight in range(2)] for numGroovePositions in range(len(groovePositions))] 
    print "\n glv********************************\n",glv
    print "\n gp********************************\n",groovePositions
    print "\n len(groovePositions)\n", len(groovePositions)
   
    
#    for i in range(len(groovePositions)):
#        print "i",i
#        processedGrooveList[0][i]=[]
#        processedGrooveList[1][i]=[]

    for i,groovePositionsItem in enumerate(groovePositions):
        print "\ni", i, groovePositionsItem,"\n" 
        processedGrooveList[i][0]=[]
        processedGrooveList[i][1]=[]
     
        leftXPosition=groovePositionsItem[0]
        rightXPosition=groovePositionsItem[1]
        for k,glvItem in enumerate(glv):
            print "\nk", k, glvItem,"\n"
            print "\nPgl", processedGrooveList
            # enumerate across slices
            for l,sliceGap in enumerate(glvItem):
                print "\nl", l, sliceGap,"\n"
                (xl,yl,zl,xr,yr,zr)=sliceGap
                if (  ( (xl+grooveXDeviation)>leftXPosition) & ( (xl-grooveXDeviation)<leftXPosition) ):
                    # falls within groove 
                    processedGrooveList[i][0].append((xl,yl,zl))
                if (  ( (xr+grooveXDeviation)>rightXPosition) & ( (xr-grooveXDeviation)<rightXPosition) ):
                    # falls within groove 
                    processedGrooveList[i][1].append((xr,yr,zr))
                
     
        
    
    print "\nProcessed Groove List\n"
    for i,processedGrooveListItem in enumerate(processedGrooveList):
        print "\n******************************Grove # ",i
        print "\n&&&&&&&&&&&&&&& left", processedGrooveListItem[0]
        print "\n&&&&&&&&&&&&&&& right", processedGrooveListItem[1]
  
    print "\n"
     
    return(processedGrooveList)
    

def CutPD(pd,pln):

    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(pln)
    cutterA.SetInputData(pd)
    cutterA.Update()
#    
    escutterA = vtk.vtkDataSetSurfaceFilter()
    escutterA.SetInputConnection(cutterA.GetOutputPort())
    escutterA.Update()

    return(escutterA.GetOutput())

def ComputePlane(p0,p1):
    
    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, 0]
    
    print "\n P0 = ", p0
    print "\n P1 = ",p1
    print "\n P2 = ", p2
#    
    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
        
    nrml=[0.0,0.0,0.0]
    p01=[0.0,0.0,0.0]
    p12=[0.0,0.0,0.0]

    math=vtk.vtkMath()
    math.Subtract(p1,p0,p01)
    math.Subtract(p2,p1,p12)
    math.Cross(p01,p12, nrml)

#    planeA=vtk.vtkPlaneSource()
#    planeA.SetPoint1(p0[0],p0[1],p0[2])
#    planeA.SetPoint2(p1[0],p1[1],p1[2])
#    planeA.SetOrigin(p2[0],p2[1],p2[2])
    
    pln=vtk.vtkPlane()
#    change Z baxck to 0
    pln.SetOrigin(p1)
    pln.SetNormal(nrml)
    
    return(pln)


def ComputeSplitPlaneFromBB(clippedOriginalMesh):
    
    bb=vtk.vtkBoundingBox()
    bb.SetInputData(clippedOriginalMesh)
    bb.GetCenter(cx,cy,cz)
    bb.Update()
    
    
    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, 0]
    
    print "\n P0 = ", p0
    print "\n P1 = ",p1
    print "\n P2 = ", p2
#    
    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
        
    nrml=[0.0,0.0,0.0]
    p01=[0.0,0.0,0.0]
    p12=[0.0,0.0,0.0]

    math=vtk.vtkMath()
    math.Subtract(p1,p0,p01)
    math.Subtract(p2,p1,p12)
    math.Cross(p01,p12, nrml)

#    planeA=vtk.vtkPlaneSource()
#    planeA.SetPoint1(p0[0],p0[1],p0[2])
#    planeA.SetPoint2(p1[0],p1[1],p1[2])
#    planeA.SetOrigin(p2[0],p2[1],p2[2])
    
    pln=vtk.vtkPlane()
#    change Z baxck to 0
    pln.SetOrigin(p1)
    pln.SetNormal(nrml)
    
    return(pln)
    


    

def FindAngleBetweenPointsAndXAxis(p0,p1):
    deltaY = p1[1]-p0[1]
    deltaX= p1[0]-p0[0]
#    deltaY = p0[1]-p1[1]
#    deltaX= p0[0]-p1[0]
    angleInDegrees = np.arctan(deltaY / deltaX) * 180.0 / np.pi
    
    print ("\n angle in degreee is ", angleInDegrees)
    return(angleInDegrees)
    
    
def FindGrooveBestFitLine3(clippedOriginalMesh,nxlow,nxhigh,nylow,nyhigh):
    
    normPoints = clippedOriginalMesh.GetPointData().GetNormals()
    numPoints = clippedOriginalMesh.GetNumberOfCells()

    # convert normal vectors to 3 scalar arrays

    data1 = vtk.vtkFloatArray()
    data1.SetNumberOfComponents(1)
    data1.SetName("xn")

    data2 = vtk.vtkFloatArray()
    data2.SetNumberOfComponents(1)
    data2.SetName("yn")

    data3 = vtk.vtkFloatArray()
    data3.SetNumberOfComponents(1)
    data3.SetName("zn")


    for i in range (0,numPoints):

        n0=normPoints.GetTuple(i)
        x,y,z=n0[:3]
        ##print "\nnormals ", x,y,z,"\n"

        data1.InsertValue(i,x)
        data2.InsertValue(i,y)
        data3.InsertValue(i,z)

  
#  itemList=[]
#    for i in range (0,numCells):
#
#        a0 = areaCells.GetTuple(i)
#        area=a0[0]
#        ##print "\narea", area,"\n"
#        n0=normCells.GetTuple(i)
#        x,y,z=n0[:3]
#        ##print "\nnormals ", x,y,z,"\n"
#
#        data1.InsertValue(i,x)
#        data2.InsertValue(i,y)
#        data3.InsertValue(i,z)
#
#        itemEntry=(i,x,y,z,area)
#        itemList.append(itemEntry)
#
#
#    itemList.sort(key=lambda tup: tup[4])


    clippedOriginalMesh.GetPointData().AddArray(data1)
    clippedOriginalMesh.GetPointData().AddArray(data2)
    clippedOriginalMesh.GetPointData().AddArray(data3)
    #clippedOriginalMesh.Update()
    
    LogVTK(clippedOriginalMesh,"c:\\temp\\beforesurfacethresh.vtp")

         
    thresholdx = vtk.vtkThreshold()
    thresholdx.SetInputData(clippedOriginalMesh)
   
    thresholdx.ThresholdBetween(0.8,1)
    #thresholdx.SetInputArrayToProcess(0, 0, 0, 1, "yn")
    thresholdx.SetInputArrayToProcess(0, 0, 0, 0, "xn")
    thresholdx.Update()
    

    esz2 = vtk.vtkDataSetSurfaceFilter()
    esz2.SetInputConnection(thresholdx.GetOutputPort())
    esz2.Update()
    
    LogVTK(esz2.GetOutput(),"c:\\temp\\surfacethresh.vtp")
    
#    pd = clippedOriginalMesh
#    fe=vtk.vtkFeatureEdges()
#    fe.SetInputData(pd)
#    fe.BoundaryEdgesOn()
#    fe.Update()
#    
    
    
#    boundaryOfGroovePD=fe.GetOutput()
#    
#    minXBound=boundaryOfGroovePD.GetBounds()[0]
#    maxXBound=boundaryOfGroovePD.GetBounds()[1]
#    
#
#    planeHalf=vtk.vtkPlane()
#    # change Z baxck to 0
#    planeHalf.SetOrigin((minXBound+maxXBound)/2.0,0,0)
#    planeHalf.SetNormal(1,0,0)
#
#    
#    rsog=ClipPDPlane(boundaryOfGroovePD,planeHalf,False)
#    LogVTK(rsog,"c:\\temp\\rightSideOfgroove.vtp")
    bfl=FindBFL1(esz2.GetOutput())
    ang=FindAngleBetweenPointsAndXAxis(bfl[0],bfl[1])
    

    
    return( (ang,bfl) )

def FindGrooveBestFitLine1(clippedOriginalMesh):
    
    pd = clippedOriginalMesh
    fe=vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.Update()
    
    boundaryOfGroovePD=fe.GetOutput()
    
    minXBound=boundaryOfGroovePD.GetBounds()[0]
    maxXBound=boundaryOfGroovePD.GetBounds()[1]
    

    planeHalf=vtk.vtkPlane()
    # change Z baxck to 0
    planeHalf.SetOrigin((minXBound+maxXBound)/2.0,0,0)
    planeHalf.SetNormal(1,0,0)

    
    rsog=ClipPDPlane(boundaryOfGroovePD,planeHalf,False)
    LogVTK(rsog,"c:\\temp\\rightSideOfgroove.vtp")
    bflr=FindBFL(rsog) 
    plnTrimRight=ComputePlane(bflr[0],bflr[1])
    grooveWithRightSideTrimmed=ClipPDPlane(clippedOriginalMesh,plnTrimRight,False)
    LogVTK(grooveWithRightSideTrimmed,"c:\\temp\\rightSideOfgrooveTrimmed.vtp")
    angr=FindAngleBetweenPointsAndXAxis(bflr[0],bflr[1])
    
    
    lsog=ClipPDPlane(boundaryOfGroovePD,planeHalf,True)
    LogVTK(lsog,"c:\\temp\\leftSideOfgroove.vtp")
    bfll=FindBFL(lsog)
    plnTrimLeft=ComputePlane(bfll[0],bfll[1])
    grooveWithBothSidesTrimmed=ClipPDPlane(grooveWithRightSideTrimmed,plnTrimLeft,True)
    LogVTK(grooveWithBothSidesTrimmed,"c:\\temp\\bothSidesOfgrooveTrimmed.vtp")
    angl=FindAngleBetweenPointsAndXAxis(bfll[0],bfll[1])
    
    result = (angr,bflr[0],bflr[1],angl, bfll[0],bfll[1],grooveWithBothSidesTrimmed)
    
    return(result)
   
    
  
#
#tireName="c:\\temp\\groove24.vtp"
##tireName="c:\\temp\\vgedges1.vtp"
#
#
#reader=vtk.vtkXMLPolyDataReader()
#reader.SetFileName(tireName )
#reader.Update()
#
#
#tire_vtk_array= reader.GetOutput().GetPoints().GetData()
#tire_numpy_array = numpy_support.vtk_to_numpy(tire_vtk_array)
#numpts=len(tire_numpy_array)
#tire_numpy_array[:,2]=np.zeros((numpts))
#print ("\n*** start")
#
##cloud = pcl.load("C:\\Projects\\tireProgramming\\pcl\\table_scene_mug_stereo_textured.pcd")
#cloud = pcl.PointCloud(tire_numpy_array)
#
#cloud_cyl = cloud
#seg = cloud_cyl.make_segmenter_normals(ksearch=50)
#seg.set_optimize_coefficients(True)
#seg.set_model_type(pcl.SACMODEL_LINE)
##seg.set_model_type(pcl.SACMODEL_CIRCLE2D)
#
#seg.set_normal_distance_weight(1)
#seg.set_method_type(pcl.SAC_RANSAC)
#seg.set_max_iterations(10000)
#seg.set_distance_threshold(0.0005)
#seg.set_radius_limits(0.29,0.34)
#indices, model = seg.segment()
#
#print(model)
#print ("\n *****")
#
#cloud_cylinder = cloud_cyl.extract(indices, negative=False)
#cloud_cylinder.to_file("C:\\Projects\\tireProgramming\\pcl\\tirelines.pcd")
#p0=[model[0],model[1],model[2]]
#p1=[model[3],model[4],model[5]]
#
#aid= ta.FindAngleBetweenPointsAndXAxis(p0,p1)
#m=(model[4]-model[1])/(model[3]-model[0])
#b=model[4]-m*model[3]
#print("\n m b", m , b)
#
#  

def FindBFL1(grooveHalf):
    tire_vtk_array= grooveHalf.GetPoints().GetData()
    tire_numpy_array = numpy_support.vtk_to_numpy(tire_vtk_array)
    

    cloud = pcl.PointCloud(tire_numpy_array)

    
    cloud_cyl = cloud
    seg = cloud_cyl.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_LINE)
    #seg.set_model_type(pcl.SACMODEL_CIRCLE2D)
    
    seg.set_normal_distance_weight(1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(10000)
    seg.set_distance_threshold(0.0005)
    #seg.set_radius_limits(0.29,0.34)
    indices, model = seg.segment()
    p0=[model[0],model[1],model[2]]
    p1=[model[3],model[4],model[5]]
#
    
    return((p0,p1))
    
      


  
def FindBFL(grooveHalf)  :  
    
    tire_vtk_array= grooveHalf.GetPoints().GetData()
    tire_numpy_array = numpy_support.vtk_to_numpy(tire_vtk_array)
    
    grooveBoundaryPointsXYZ=tire_numpy_array


    numpts=len(grooveBoundaryPointsXYZ)
    

    pointsArray=grooveBoundaryPointsXYZ[:,0:2]

    print "\nPoints Arrayt", pointsArray
    model = LineModel()
   
    
    if  (pointsArray.size > 0):
        model.estimate(pointsArray)
    

        model_robust, inliers = ransac(pointsArray, LineModel, min_samples=10,
                    residual_threshold=0.001, max_trials=1000)                               
#        print "\ninliers",inliers
#        print "\line model params",   model_robust.params,"\n"    
    
    icount=0
    print "\n&&&&&&&&&&&&& Inliers\n"
    for i in range(numpts):
        if (inliers[i]):
            icount=icount+1
#            print "\n",pointsArray[i,0],pointsArray[i,1]
            
    
    print "\nicount = ",icount
    avgZ=np.average(grooveBoundaryPointsXYZ[inliers][:,2])
    
    # find Z to map back to
#    z=0
#    numInliers=0
#    for i,item in enumerate(inliers):
#        if item:
#            numInliers=numInliers+1
#            print "\n z to account for", i, grooveList[i][2]
#            z=z+grooveList[i][2]
#    avgZ=z/numInliers
#    print "\n avg z ", avgZ
    
  
    
    print model_robust.params
    
    #bbox=[0,0,0,0,0,0]
    #line_x = np.zeros(2)
    #line_y = np.arange(0.01,0.065,0.005)
    line_y = np.arange(np.min(grooveBoundaryPointsXYZ[:,1]),np.max(grooveBoundaryPointsXYZ[:,1]),0.005)
    line_x_robust=model_robust.predict_x(line_y)
    
    #Create two points, P0 and P1
#    p0 = [line_y[0],line_x_robust[0],avgZ]
#    p1 = [line_y[1],line_x_robust[1],avgZ]
#    
    #Create two points, P0 and P1
    p0 = [line_x_robust[0],line_y[0],avgZ]
    p1 = [line_x_robust[10],line_y[len(line_y)-1],avgZ]
    
    print "\n %%%%%%% Line Y robust", line_x_robust[0], line_x_robust[10]
    print "\n %%%%%%% Line Y robust", line_x_robust 
    
    return((p0,p1))
    
#
#    
#    lineSource = vtk.vtkLineSource()
#    lineSource.SetPoint1(p0);
#    lineSource.SetPoint2(p1)
#    lineSource.Update();
#    
#    LogVTK(lineSource.GetOutput(), "c:\\temp\\grooveline.vtp")
##    writer_td53 = vtk.vtkXMLPolyDataWriter()
##    #writer = vtk.vtkSimplePointsWriter()
##    filename=directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp"
##
##    writer_td53.SetInputData(lineSource.GetOutput())
##    writer_td53.SetFileName(filename)
##    writer_td53.Write()
#
#    print "\nyy line y length", len(line_y)
#    
#    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, avgZ+0.001]
#    
#    print "\n P0 = ", p0
#    print "\n P1 = ",p1
#    print "\n P2 = ", p2
##    
##    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
##    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
#    
#        
#    
#    nrml=[0.0,0.0,0.0]
#    p01=[0.0,0.0,0.0]
#    p12=[0.0,0.0,0.0]
#
#    math=vtk.vtkMath()
#    math.Subtract(p1,p0,p01)
#    math.Subtract(p2,p1,p12)
#    math.Cross(p01,p12, nrml)
#    
#    
#
#    
#    
#    planeA=vtk.vtkPlaneSource()
#    planeA.SetPoint1(p0[0],p0[1],p0[2])
#    planeA.SetPoint2(p1[0],p1[1],p1[2])
#    planeA.SetOrigin(p2[0],p2[1],p2[2])
#    
#
#    planeA.Update()
#    
#    filename="c:\\temp\plane1.vtp"
#    LogVTK(planeA.GetOutput(),filename)
#    
#    
#
##
##    writer_td63 = vtk.vtkXMLPolyDataWriter()
##    writer_td63.SetInputData(planeA.GetOutput())
##    filename="c:\\temp\plane" + grooveLineLabel +".vtp"
##    writer_td63.SetFileName(filename)
##    writer_td63.Write()
##    
#    planeA=vtk.vtkPlane()
#    # change Z baxck to 0
#    planeA.SetOrigin(p1)
#    planeA.SetNormal(nrml)
#    cutterA=vtk.vtkCutter()
#    cutterA.SetSortBy(0)
#    cutterA.SetCutFunction(planeA)
#    cutterA.SetInputData(clippedOriginalMesh)
#    cutterA.Update()
##    
#    escutterA = vtk.vtkDataSetSurfaceFilter()
#    escutterA.SetInputConnection(cutterA.GetOutputPort())
#    escutterA.Update()
#    
#    filename="c:\\temp\grooveclippedbyplane.vtp"
#    LogVTK(escutterA.GetOutput(),filename)
#    
#    res=ClipPDPlane(clippedOriginalMesh,planeA,False)
#    LogVTK(res,"c:\\temp\\gcp.vtp")
##    writer_td63 = vtk.vtkXMLPolyDataWriter()
##    writer_td63.SetInputData(escutterA.GetOutput())
##    filename=directoryForLog+"cutter" + grooveLineLabel +".vtp"
##    writer_td63.SetFileName(filename)
##    writer_td63.Write()
##    
##    print "\nnrml", nrml
##    print "\norigin", p1
#    
#    deltaY = p1[1]-p0[1]
#    deltaX= p1[0]-p0[0]
#    angleInDegrees = np.arctan((deltaY / deltaX) * 180.0 / np.pi)
#    
#    print ("\n angle in degreee is ", angleInDegrees)
#    
#
#    
#    return((p0,p1))
#    
def FindGrooveBestFitLine2(clippedOriginalMesh):
    
    pd = clippedOriginalMesh
    fe=vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.Update()
    
    tire_vtk_array= fe.GetOutput().GetPoints().GetData()
    tire_numpy_array = numpy_support.vtk_to_numpy(tire_vtk_array)
    
    grooveBoundaryPointsXYZ=tire_numpy_array

  #pointsArray = np.zeros((2, numpts))

#    print "\n && Entering findGrooveBestFitLine"
#    print "\n", grooveLineLabel
#    print "\n ", grooveList

    numpts=len(grooveBoundaryPointsXYZ)
    

    pointsArray=grooveBoundaryPointsXYZ[:,0:2]
   
#    for i,item in enumerate(grooveList):
#        print "\n item", item, "\n"
#        pointsArray[i,0] = item[0]
#        pointsArray[i,1] = item[1]
##        pointsArray[2,i] = z
    
    print "\nPoints Arrayt", pointsArray
    model = LineModel()
   
    
    if  (pointsArray.size > 0):
        model.estimate(pointsArray)
        
        
        #x = np.arange(-200, 200)
        #y = 0.2 * x + 20
        #data = np.column_stack([x, y])
        model_robust, inliers = ransac(pointsArray, LineModel, min_samples=10,
                    residual_threshold=0.0001, max_trials=1000)                               
        print "\ninliers",inliers
        print "\line model params",   model_robust.params,"\n"    
    
    icount=0
    print "\n&&&&&&&&&&&&& Inliers\n"
    for i in range(numpts):
        if (inliers[i]):
            icount=icount+1
            print "\n",pointsArray[i,0],pointsArray[i,1]
            
    
    print "\nicount = ",icount
    avgZ=np.average(grooveBoundaryPointsXYZ[inliers][:,2])
    
    # find Z to map back to
#    z=0
#    numInliers=0
#    for i,item in enumerate(inliers):
#        if item:
#            numInliers=numInliers+1
#            print "\n z to account for", i, grooveList[i][2]
#            z=z+grooveList[i][2]
#    avgZ=z/numInliers
#    print "\n avg z ", avgZ
    
  
    
    print model_robust.params
    
    #bbox=[0,0,0,0,0,0]
    #line_x = np.zeros(2)
    #line_y = np.arange(0.01,0.065,0.005)
    line_y = np.arange(np.min(grooveBoundaryPointsXYZ[:,1]),np.max(grooveBoundaryPointsXYZ[:,1]),0.005)
    line_x_robust=model_robust.predict_x(line_y)
    
    #Create two points, P0 and P1
#    p0 = [line_y[0],line_x_robust[0],avgZ]
#    p1 = [line_y[1],line_x_robust[1],avgZ]
#    
    #Create two points, P0 and P1
    p0 = [line_x_robust[0],line_y[0],avgZ]
    p1 = [line_x_robust[10],line_y[len(line_y)-1],avgZ]
    
    print "\n %%%%%%% Line Y robust", line_x_robust[0], line_x_robust[10]
    print "\n %%%%%%% Line Y robust", line_x_robust 
    

    
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(p0);
    lineSource.SetPoint2(p1)
    lineSource.Update();
    
    LogVTK(lineSource.GetOutput(), "c:\\temp\\grooveline.vtp")
#    writer_td53 = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    filename=directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp"
#
#    writer_td53.SetInputData(lineSource.GetOutput())
#    writer_td53.SetFileName(filename)
#    writer_td53.Write()

    print "\nyy line y length", len(line_y)
    
    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, avgZ+0.001]
    
    print "\n P0 = ", p0
    print "\n P1 = ",p1
    print "\n P2 = ", p2
#    
#    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
#    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
    
        
    
    nrml=[0.0,0.0,0.0]
    p01=[0.0,0.0,0.0]
    p12=[0.0,0.0,0.0]

    math=vtk.vtkMath()
    math.Subtract(p1,p0,p01)
    math.Subtract(p2,p1,p12)
    math.Cross(p01,p12, nrml)
    
    

    
    
    planeA=vtk.vtkPlaneSource()
    planeA.SetPoint1(p0[0],p0[1],p0[2])
    planeA.SetPoint2(p1[0],p1[1],p1[2])
    planeA.SetOrigin(p2[0],p2[1],p2[2])
    

    planeA.Update()
    
    filename="c:\\temp\plane1.vtp"
    LogVTK(planeA.GetOutput(),filename)
    
    

#
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(planeA.GetOutput())
#    filename="c:\\temp\plane" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
    planeA=vtk.vtkPlane()
    # change Z baxck to 0
    planeA.SetOrigin(p1)
    planeA.SetNormal(nrml)
    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(planeA)
    cutterA.SetInputData(clippedOriginalMesh)
    cutterA.Update()
#    
    escutterA = vtk.vtkDataSetSurfaceFilter()
    escutterA.SetInputConnection(cutterA.GetOutputPort())
    escutterA.Update()
    
    filename="c:\\temp\grooveclippedbyplane.vtp"
    LogVTK(escutterA.GetOutput(),filename)
    
    res=ClipPDPlane(clippedOriginalMesh,planeA,False)
    LogVTK(res,"c:\\temp\\gcp.vtp")
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(escutterA.GetOutput())
#    filename=directoryForLog+"cutter" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
#    print "\nnrml", nrml
#    print "\norigin", p1
    
    deltaY = p1[1]-p0[1]
    deltaX= p1[0]-p0[0]
    angleInDegrees = np.arctan((deltaY / deltaX) * 180.0 / np.pi)
    
    print ("\n angle in degreee is ", angleInDegrees)
    

    
    return((p0,p1))
    
def FindGrooveBestFitLine(grooveLineLabel,grooveList,bbox,clippedOriginalMesh):
  #pointsArray = np.zeros((2, numpts))

    print "\n && Entering findGrooveBestFitLine"
    print "\n", grooveLineLabel
    print "\n ", grooveList

    numpts=len(grooveList)
    pointsArray = np.zeros(( numpts,2))
    
   
    for i,item in enumerate(grooveList):
        print "\n item", item, "\n"
        pointsArray[i,0] = item[0]
        pointsArray[i,1] = item[1]
#        pointsArray[2,i] = z
    
    print "\nPoints Arrayt", pointsArray
    model = LineModel()
   
    
    if  (pointsArray.size > 0):
        model.estimate(pointsArray)
        
        
        #x = np.arange(-200, 200)
        #y = 0.2 * x + 20
        #data = np.column_stack([x, y])
        model_robust, inliers = ransac(pointsArray, LineModel, min_samples=10,
                    residual_threshold=0.0001, max_trials=1000)                               
        print "\ninliers",inliers
        print "\line model params",   model_robust.params,"\n"    
    
    icount=0
    print "\n&&&&&&&&&&&&& Inliers\n"
    for i in range(numpts):
        if (inliers[i]):
            icount=icount+1
            print "\n",pointsArray[i,0],pointsArray[i,1]
            
    
    print "\nicount = ",icount
    
    # find Z to map back to
    z=0
    numInliers=0
    for i,item in enumerate(inliers):
        if item:
            numInliers=numInliers+1
            print "\n z to account for", i, grooveList[i][2]
            z=z+grooveList[i][2]
    avgZ=z/numInliers
    print "\n avg z ", avgZ
    
  
    
    print model_robust.params
    
    #line_x = np.zeros(2)
    #line_y = np.arange(0.01,0.065,0.005)
    line_y = np.arange(bbox[2],bbox[3],0.005)
    line_x_robust=model_robust.predict_x(line_y)
    
    #Create two points, P0 and P1
#    p0 = [line_y[0],line_x_robust[0],avgZ]
#    p1 = [line_y[1],line_x_robust[1],avgZ]
#    
    #Create two points, P0 and P1
    p0 = [line_x_robust[0],line_y[0],avgZ]
    p1 = [line_x_robust[10],line_y[len(line_y)-1],avgZ]
    
    print "\n %%%%%%% Line Y robust", line_x_robust[0], line_x_robust[10]
    print "\n %%%%%%% Line Y robust", line_x_robust 
    

    
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(p0);
    lineSource.SetPoint2(p1)
    lineSource.Update();
    
    LogVTK(lineSource.GetOutput(), directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp")
#    writer_td53 = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    filename=directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp"
#
#    writer_td53.SetInputData(lineSource.GetOutput())
#    writer_td53.SetFileName(filename)
#    writer_td53.Write()

    print "\nyy line y length", len(line_y)
    
    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, avgZ+0.001]
    
    print "\n P0 = ", p0
    print "\n P1 = ",p1
    print "\n P2 = ", p2
#    
#    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
#    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
    
        
    
    nrml=[0.0,0.0,0.0]
    p01=[0.0,0.0,0.0]
    p12=[0.0,0.0,0.0]

    math=vtk.vtkMath()
    math.Subtract(p1,p0,p01)
    math.Subtract(p2,p1,p12)
    math.Cross(p01,p12, nrml)
    
    

#    
#    
#    planeA=vtk.vtkPlaneSource()
#    planeA.SetPoint1(p0[0],p0[1],p0[2])
#    planeA.SetPoint2(p1[0],p1[1],p1[2])
#    planeA.SetOrigin(p2[0],p2[1],p2[2])
#    
#
#    planeA.Update()
#    
#
#
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(planeA.GetOutput())
#    filename="c:\\temp\plane" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
    planeA=vtk.vtkPlane()
    # change Z baxck to 0
    planeA.SetOrigin(p1)
    planeA.SetNormal(nrml)
    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(planeA)
    cutterA.SetInputConnection(clippedOriginalMesh.GetOutputPort())
    cutterA.Update()
    
    escutterA = vtk.vtkDataSetSurfaceFilter()
    escutterA.SetInputConnection(cutterA.GetOutputPort())
    escutterA.Update()
    
    LogVTK(escutterA.GetOutput(), directoryForLog+"cutter" + grooveLineLabel +".vtp")
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(escutterA.GetOutput())
#    filename=directoryForLog+"cutter" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
#    print "\nnrml", nrml
#    print "\norigin", p1
    
    return((p0,p1))
    


def FindGrooveBestFitCircle(grooveLineLabel,grooveList,bbox,clippedOriginalMesh):
  #pointsArray = np.zeros((2, numpts))

    print "\n && Entering findGrooveBestFitCircle"
    print "\n", grooveLineLabel
    print "\n ", grooveList

    numpts=len(grooveList)
    pointsArray = np.zeros(( numpts,2))
    
   
    for i,item in enumerate(grooveList):
        #print "\n item", item, "\n"
        pointsArray[i,0] = item[0]
        pointsArray[i,1] = item[1]
#        pointsArray[2,i] = z
    
    print "\nPoints Arrayt", pointsArray
    model = CircleModel()
   
    
    if  (pointsArray.size > 0):
        model.estimate(pointsArray)
        
        
        #x = np.arange(-200, 200)
        #y = 0.2 * x + 20
        #data = np.column_stack([x, y])
        model_robust, inliers = ransac(pointsArray, LineModel, min_samples=10,
                    residual_threshold=0.0001, max_trials=1000)                               
        print "\ninliers",inliers
        print "\line model params",   model_robust.params,"\n"    
    
    icount=0
    print "\n&&&&&&&&&&&&& Inliers\n"
    for i in range(numpts):
        if (inliers[i]):
            icount=icount+1
            print "\n",pointsArray[i,0],pointsArray[i,1]
            
    
    print "\nicount = ",icount
    
    # find Z to map back to
    z=0
    numInliers=0
    for i,item in enumerate(inliers):
        if item:
            numInliers=numInliers+1
            print "\n z to account for", i, grooveList[i][2]
            z=z+grooveList[i][2]
    avgZ=z/numInliers
    print "\n avg z ", avgZ
    
  
    
    print model_robust.params
    
    #line_x = np.zeros(2)
    #line_y = np.arange(0.01,0.065,0.005)
    line_y = np.arange(bbox[2],bbox[3],0.005)
    line_x_robust=model_robust.predict_x(line_y)
    
    #Create two points, P0 and P1
#    p0 = [line_y[0],line_x_robust[0],avgZ]
#    p1 = [line_y[1],line_x_robust[1],avgZ]
#    
    #Create two points, P0 and P1
    p0 = [line_x_robust[0],line_y[0],avgZ]
    p1 = [line_x_robust[10],line_y[len(line_y)-1],avgZ]
    
    print "\n %%%%%%% Line Y robust", line_x_robust[0], line_x_robust[10]
    print "\n %%%%%%% Line Y robust", line_x_robust 
    

    
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(p0);
    lineSource.SetPoint2(p1)
    lineSource.Update();
    
    LogVTK(lineSource.GetOutput(), directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp")
#    writer_td53 = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    filename=directoryForLog+"grooveLine"+str(grooveLineLabel)+".vtp"
#
#    writer_td53.SetInputData(lineSource.GetOutput())
#    writer_td53.SetFileName(filename)
#    writer_td53.Write()

    print "\nyy line y length", len(line_y)
    
    p2 = [ (p0[0]+p1[0])/2,(p0[1]+p1[1])/2, avgZ+0.001]
    
    print "\n P0 = ", p0
    print "\n P1 = ",p1
    print "\n P2 = ", p2
#    
#    p01=[ (p0[0]-p1[0]),(p0[1]-p1[1]), (p0[2]-p1[2]) ]
#    p12=[ (p1[0]-p2[0]),(p1[1]-p2[1]), (p1[2]-p2[2]) ]
    
        
    
    nrml=[0.0,0.0,0.0]
    p01=[0.0,0.0,0.0]
    p12=[0.0,0.0,0.0]

    math=vtk.vtkMath()
    math.Subtract(p1,p0,p01)
    math.Subtract(p2,p1,p12)
    math.Cross(p01,p12, nrml)
    
    

#    
#    
#    planeA=vtk.vtkPlaneSource()
#    planeA.SetPoint1(p0[0],p0[1],p0[2])
#    planeA.SetPoint2(p1[0],p1[1],p1[2])
#    planeA.SetOrigin(p2[0],p2[1],p2[2])
#    
#
#    planeA.Update()
#    
#
#
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(planeA.GetOutput())
#    filename="c:\\temp\plane" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
    planeA=vtk.vtkPlane()
    # change Z baxck to 0
    planeA.SetOrigin(p1)
    planeA.SetNormal(nrml)
    cutterA=vtk.vtkCutter()
    cutterA.SetSortBy(0)
    cutterA.SetCutFunction(planeA)
    cutterA.SetInputConnection(clippedOriginalMesh.GetOutputPort())
    cutterA.Update()
    
    escutterA = vtk.vtkDataSetSurfaceFilter()
    escutterA.SetInputConnection(cutterA.GetOutputPort())
    escutterA.Update()
    
    LogVTK(escutterA.GetOutput(), directoryForLog+"cutter" + grooveLineLabel +".vtp")
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    writer_td63.SetInputData(escutterA.GetOutput())
#    filename=directoryForLog+"cutter" + grooveLineLabel +".vtp"
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
#    
#    print "\nnrml", nrml
#    print "\norigin", p1
    
    return((p0,p1))
   
def ProcessGroove(grooveNumber, grooveListLeft, grooveListRight,bbox):
    grooveDepthConstant=0.080
    (p0L,p1L) = findGrooveBestFitLine("Left",grooveListLeft,bbox) 
    (p0R,p1R) = findGrooveBestFitLine("Right",grooveListRight,bbox) 
    gb=vtk.vtkCubeSource()
    gb.SetBounds(p0L[0], p0R[0], bbox[2], bbox[3],  p0L[2]-grooveDepthConstant, p0L[2]+0.005)
    
    gb.Update()

    
  
    
    LogVTK(gb.GetOutput(), directoryForLog+"grooveBox_"+str(grooveNumber)+".vtp")
#    writer_td63 = vtk.vtkXMLPolyDataWriter()
#    #writer = vtk.vtkSimplePointsWriter()
#    filename=directoryForLog+"grooveBox_"+str(grooveNumber)+".vtp"
#    writer_td63.SetInputData(gb.GetOutput())
#    writer_td63.SetFileName(filename)
#    writer_td63.Write()
    return(gb)

def ExploitSymmetryToImproveOrientation(meshImprove):
    # rotate mesh 180
    
    # run ICP on mesh 
    # return transform and error
    print "\n && Entering ExploitSymmetryToImproveOrientation"
    
    meshImproveRotate180Transform=vtk.vtk.vtkTransform()
#    meshImproveRotate180Transform.PostMultiply()
    meshImproveRotate180Transform.RotateZ(180)
    meshImproveRotate180Transform.Update()
    
    
    meshImproveRotate180 = vtk.vtkTransformPolyDataFilter()
    meshImproveRotate180.SetInputConnection(meshImprove.GetOutputPort())
    
    meshImproveRotate180.SetTransform(meshImproveRotate180Transform)
    meshImproveRotate180.Update()
    
    esmeshImproveRotate180 = vtk.vtkDataSetSurfaceFilter()
    esmeshImproveRotate180.SetInputConnection(meshImproveRotate180.GetOutputPort())
    esmeshImproveRotate180.Update()
    
    # as a sanity check, compute the avy distance before ICP
    #print "************* Computing avg distance before ICP  *************"
    dontcare,avgdist=ComputeDistance(7777,meshImprove,esmeshImproveRotate180,0 , False )  
    print "\n ************* Computing avg distance before ICP  *************", avgdist
    
    icp = vtk.vtkIterativeClosestPointTransform()

    icp.SetSource(esmeshImproveRotate180.GetOutput())
    icp.SetTarget(meshImprove.GetOutput())
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToRMS();
    #icp.DebugOn()
    #icp.SetMaximumMeanDistance      ( 0.002 )
    icp.SetMaximumNumberOfIterations(20000)
    
    #icp.SetMaximumNumberOfIterations(4)
    #icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    
    print icp.GetMeanDistance()
    
    print icp.GetLandmarkTransform()
    
    print icp.GetLandmarkTransform().GetMatrix()
    
    TransformToFile(icp, directoryForLog+"icpTransform.txt")    
    
    m0=vtk.vtkMatrix4x4()
    m1=vtk.vtkMatrix4x4()
    m2=vtk.vtkMatrix4x4()
  
    m1=icp.GetMatrix()
    m0.Invert(m1,m2)

   
    icpInvTransform =vtk.vtkTransform()  
    icpInvTransform.SetMatrix(m2)
    
    TransformToFile(icpInvTransform, directoryForLog+"icpInvTransform.txt")    


    
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(esmeshImproveRotate180.GetOutput())
    
    icpTransformFilter.SetTransform(icp)
    #icpTransformFilter.SetTransform(icp)
    
    icpTransformFilter.Update()
    
    #transformedSource = icpTransformFilter.GetOutput()
    
    fullyOrientedmesh=TransformPD(meshImprove.GetOutput(),icpInvTransform)
    LogVTK(fullyOrientedmesh.GetOutput(), directoryForLog+"mesh_after_pcaandicp.vtp")
    LogVTK(icpTransformFilter.GetOutput(), directoryForLog+"rotated_mesh_after_pcaandicp.vtp")
#    writer_td60 = vtk.vtkXMLPolyDataWriter()
#    writer_td60.SetInputData(icpTransformFilter.GetOutput())
#    writer_td60.SetFileName(directoryForLog+"rotated_mesh_after_pcaandicp.vtp")
#    writer_td60.Write()
      
    dontcare,avgdist=ComputeDistance(7777,meshImprove,TransformPD(esmeshImproveRotate180.GetOutput(), icp),0 , True )    
    
    print "\n $$$$$$$$$$$$$$$$$$$$$$$$ avg dist is ", avgdist 
    
    return(icp)

def PrepareForICPByCorrectingForYPositionA(allmesh_clipped_to_grooves_oriented):
    

    mesh1 = allmesh_clipped_to_grooves_oriented
    
    
    meshImproveRotate180Transform=vtk.vtk.vtkTransform()
    meshImproveRotate180Transform.PostMultiply()
    meshImproveRotate180Transform.RotateZ(180)
    
    mesh1_180 =TransformPD(mesh1,meshImproveRotate180Transform).GetOutput()
    
#    dontcare,avgdist=ComputeDistance(777,mesh1,mesh1_180,0, True)
    
    boundmesh1=mesh1.GetBounds()
    boundmesh1_180=mesh1_180.GetBounds()
    
    if (boundmesh1[0]<boundmesh1_180[0]):
        yDirection=1
    else:
        yDirection=-1
    
    yAmountPerStep=0.0005
    
    roughAlignmentTransform=vtk.vtk.vtkTransform()
    roughAlignmentTransform.Translate(0,yAmountPerStep*yDirection,0)
    #
    reached_min_dist=False
    moved_mesh=mesh1
    prev_avgdist=99999
    translatePos=0
    while(not reached_min_dist):
        dontcare, avgdist=ComputeDistance(777,moved_mesh,mesh1_180,0, True)
     
        print "\n avg dist and translatePos are", avgdist, translatePos
        if (prev_avgdist<avgdist):
            reached_min_dist=True
        else:
            translatePos=translatePos+yAmountPerStep*yDirection
            roughAlignmentTransform.Translate(translatePos,0,0)
            moved_mesh=TransformPD(mesh1,roughAlignmentTransform)
        
        prev_avgdist=avgdist
        prev_translatePos=translatePos
        
    #     move mesh1 up(down) until dist is minimized
    
    prev_translatePos=0.0030    
    current_transform = vtk.vtk.vtkTransform()
    current_transform.Translate(prev_translatePos/2.0,0,0)
    newmesh1=TransformPD(mesh1,current_transform)
    newmesh1_180 = TransformPD(newmesh1, meshImproveRotate180Transform)
    dontcare,avgdist=ComputeDistance(777,newmesh1, newmesh1_180,0, False)
    
    #print "\n 88888888888 avg Distance is ", avgdist
    
    return(reached_min_dist,current_transform)
        
def PrepareForICPByCorrectingForYPosition(allmesh_clipped_to_grooves_oriented):
    
    
    # this logic is wrong - need to move halfway up and halfway down

    mesh1 = allmesh_clipped_to_grooves_oriented
    
    
    meshImproveRotate180Transform=vtk.vtk.vtkTransform()
    meshImproveRotate180Transform.PostMultiply()
    meshImproveRotate180Transform.RotateZ(180)
    
    mesh1_180 =TransformPD(mesh1.GetOutput(),meshImproveRotate180Transform)
    
#    dontcare,avgdist=ComputeDistance(777,mesh1,mesh1_180,0, True)
    
    boundmesh1=[0,0,0,0,0,0]
    boundmesh1=mesh1.GetOutput().GetBounds()
    boundmesh1_180=[0,0,0,0,0,0]
    boundmesh1_180=mesh1_180.GetOutput().GetBounds()
    
    if (boundmesh1[2]<boundmesh1_180[2]):
        yDirection=1
    else:
        yDirection=-1
        
    
    # force to 1 for now  
    yDirection=1
    
    #yAmountPerStep=0.001
    # force it to 0.007 for now
    yAmountPerStep=-0.007
    
    roughAlignmentTransform=vtk.vtk.vtkTransform()
    roughAlignmentTransform.Translate(0,yAmountPerStep*yDirection,0)
    #
    reached_min_dist=False
    moved_mesh=mesh1
    prev_avgdist=99999
    translatePos=0
    while(not reached_min_dist):
        dontcare, avgdist=ComputeDistance(777,moved_mesh,mesh1_180,0, True)
     
        print "\n avg dist and translatePos are", avgdist, translatePos
        if (prev_avgdist<avgdist):
            reached_min_dist=True
        else:
            translatePos=translatePos+yAmountPerStep*yDirection
            roughAlignmentTransform.Translate(0,translatePos,0)
            moved_mesh=TransformPD(mesh1.GetOutput(),roughAlignmentTransform)
        
        prev_avgdist=avgdist
        prev_translatePos=translatePos
        
    #     move mesh1 to the right until dist is minimized
    
    prev_translatePos=0.0030    
    current_transform = vtk.vtk.vtkTransform()
    current_transform.Translate(0,prev_translatePos/2.0,0)
    newmesh1=TransformPD(mesh1.GetOutput(),current_transform)
    newmesh1_180 = TransformPD(newmesh1.GetOutput(), meshImproveRotate180Transform)
    dontcare,avgdist=ComputeDistance(777,newmesh1, newmesh1_180,0, False)
    
    #print "\n 88888888888 avg Distance is ", avgdist
    
    return(reached_min_dist,current_transform)
        


def PrepareForICPByCorrectingForXPosition(allmesh_clipped_to_grooves_oriented):
    
#    filename="C:\\Temp\\m1.ply_4985752_\\allmesh_oriented_.vtp"
#    mesh1=vtk.vtkXMLPolyDataReader()
#    mesh1.SetFileName(filename)
#    mesh1.Update()
#    
#    cln_mesh1=vtk.vtkCleanPolyData()
#    cln_mesh1.SetInputData(mesh1.GetOutput())
#    cln_mesh1.SetTolerance(0.01)
#    cln_mesh1.Update()
#    
#    #  don't dumb down resoutuion
#    cln_mesh1=mesh1
    
    # correct the original clipped mesh to account for PCA
    mesh1 = allmesh_clipped_to_grooves_oriented
    
    
    meshImproveRotate180Transform=vtk.vtk.vtkTransform()
    meshImproveRotate180Transform.PostMultiply()
    meshImproveRotate180Transform.RotateZ(180)
    
    mesh1_180 =TransformPD(mesh1.GetOutput(),meshImproveRotate180Transform)
    
#    dontcare,avgdist=ComputeDistance(777,mesh1,mesh1_180,0, True)
    
    boundmesh1=[0,0,0,0,0,0]
    boundmesh1=mesh1.GetOutput().GetBounds()
    boundmesh1_180=[0,0,0,0,0,0]
    boundmesh1_180=mesh1_180.GetOutput().GetBounds()
    
    if (boundmesh1[0]<boundmesh1_180[0]):
        xDirection=1
    else:
        xDirection=-1
    
    xAmountPerStep=0.0005
    
    roughAlignmentTransform=vtk.vtk.vtkTransform()
    roughAlignmentTransform.Translate(xAmountPerStep*xDirection,0,0)
    #
    reached_min_dist=False
    moved_mesh=mesh1
    prev_avgdist=99999
    translatePos=0
    while(not reached_min_dist):
        dontcare, avgdist=ComputeDistance(777,moved_mesh,mesh1_180,0, True)
     
        print "\n avg dist and translatePos are", avgdist, translatePos
        if (prev_avgdist<avgdist):
            reached_min_dist=True
        else:
            translatePos=translatePos+xAmountPerStep*xDirection
            roughAlignmentTransform.Translate(translatePos,0,0)
            moved_mesh=TransformPD(mesh1.GetOutput(),roughAlignmentTransform)
        
        prev_avgdist=avgdist
        prev_translatePos=translatePos
        
    #     move mesh1 to the right until dist is minimized
    
    prev_translatePos=0.0030    
    current_transform = vtk.vtk.vtkTransform()
    current_transform.Translate(prev_translatePos/2.0,0,0)
    newmesh1=TransformPD(mesh1.GetOutput(),current_transform)
    newmesh1_180 = TransformPD(newmesh1.GetOutput(), meshImproveRotate180Transform)
    dontcare,avgdist=ComputeDistance(777,newmesh1, newmesh1_180,0, False)
    
    #print "\n 88888888888 avg Distance is ", avgdist
    
    return(current_transform)
        



def FindXLocationOfXNormals(pd,xnValue,interval):

    bbox=pd.GetBounds()
    bboxXRange=(bbox[1]-bbox[0])
    xspan=bboxXRange/interval
    startX=bbox[0]
    xnIntervalList=[]
    
    pdNormals = vtk.vtkPolyDataNormals()

    pdNormals.SetInputData(pd)
    pdNormals.ComputePointNormalsOn()
    pdNormals.SetFeatureAngle (23)
    pdNormals.SplittingOn ()
    pdNormals.Update()
    
    pd=pdNormals.GetOutput()

    
    for i in range(interval):
        endX=startX+xspan
        
        bboxTest=(startX,endX,bbox[2],bbox[3],bbox[4],bbox[5])
        
        clippedToSpan=Clip(pd,bboxTest)
        xnarray = clippedToSpan.GetOutput().GetPointData().GetArray("Normals")
        
        sumxn=0.0
        for j in range(clippedToSpan.GetOutput().GetNumberOfPoints()):
            val=xnarray.GetTuple3(j)[0]
            sumxn=sumxn+val
        
        print ("\n xnA sum ", i, sumxn,  float(sumxn/j) )
        startX=endX
        avgXNorm=float(sumxn/j)
        xnIntervalList.append(avgXNorm)
        if (avgXNorm>-xnValue):
            ilow=i
            break
    
    endX=bbox[1]
    for i in range(interval-1,0,-1):
        startX=endX-xspan
        
        bboxTest=(startX,endX,bbox[2],bbox[3],bbox[4],bbox[5])
        
        clippedToSpan=Clip(pd,bboxTest)
        xnarray = clippedToSpan.GetOutput().GetPointData().GetArray("Normals")
        
        sumxn=0.0
        for j in range(clippedToSpan.GetOutput().GetNumberOfPoints()):
            val=xnarray.GetTuple3(j)[0]
            sumxn=sumxn+val
        
        print ("\n xnB sum ", i, sumxn,  float(sumxn/j) )
        endX=startX
        avgXNorm=float(sumxn/j)
        xnIntervalList.append(avgXNorm)
        if (avgXNorm<xnValue):
            ihigh=i
            break
        
    return( [ bbox[0]+float(ilow)*xspan,bbox[0]+float(ihigh)*xspan,bbox[2],bbox[3],bbox[4],bbox[5] ] )
    #return(ilow,ihigh)
            

def FindXLocationOfXNormals1(pd,xnValue,interval):

    bbox=pd.GetBounds()
    bboxXRange=(bbox[1]-bbox[0])
    xspan=bboxXRange/interval
    startX=bbox[0]
    xnIntervalList=[]
    
    for i in range(interval):
        endX=startX+xspan
        
        bboxTest=(startX,endX,bbox[2],bbox[3],bbox[4],bbox[5])
        
        clippedToSpan=Clip(pd,bboxTest)
        normals = clippedToSpan.GetOutput().GetPointData().GetArray("Normals")
        
        sumxn=0.0
        for j in range(clippedToSpan.GetOutput().GetNumberOfPoints()):
            val=normals.GetTuple3(j)[0]
            sumxn=sumxn+val
        
        print ("\n xn sum ", i, sumxn,  float(sumxn/j) )
        startX=endX
        xnIntervalList.append(float(sumxn/j))
    
    
    xnIntervalListSort=sorted(xnIntervalList)
    
    for i in range(len(xnIntervalListSort)):
        if xnIntervalList[i]>-xnValue:
            break
    ilow=i
    for i in range(len(xnIntervalListSort)-1,0,-1):
        if xnIntervalList[i]<xnValue:
            break
    ihigh=i
    
    return(ilow,ihigh)
    
def RenameArrayAndSaveFile(sourceFilePathName, destFilePathName, sourceArrayName, destArrayName, pointArrayOnly):

#dirToTest="c:\\temp\\del_2016-07-04_23-30-13"
#filename=dirToTest+"\\"+ "distcurv"  + ".vtp"


    reader = vtk.vtkXMLPolyDataReader()
    #reader = vtk.vtkPLYReader()
    reader.SetFileName(sourceFilePathName)
    reader.Update()


    distPIndex=GetArrayByName(reader.GetOutput(), sourceArrayName)
    distArrayPoint= reader.GetOutput().GetPointData().GetArray(distPIndex)
    distArrayPoint.SetName(destArrayName)

    if not(pointArrayOnly):

        distCIndex=GetArrayByNamePC(reader.GetOutput(),sourceArrayName,False)
        distArrayCell= reader.GetOutput().GetCellData().GetArray(distCIndex)
        distArrayCell.SetName(destArrayName)


    pdcopy= vtk.vtkPolyData()
    pdcopy.ShallowCopy(reader.GetOutput())

    pdcopy.GetPointData().AddArray(distArrayPoint)
    pdcopy.GetCellData().AddArray(distArrayCell)


    LogVTK(pdcopy,destFilePathName)

def RenameArray(pd,sourceArrayName, destArrayName, pointArrayOnly):
    
    distPIndex=GetArrayByName(pd, sourceArrayName)
    distArrayPoint= pd.GetPointData().GetArray(distPIndex)
    distArrayPoint.SetName(destArrayName)

    if not(pointArrayOnly):

        distCIndex=GetArrayByNamePC(pd,sourceArrayName,False)
        distArrayCell= pd.GetCellData().GetArray(distCIndex)
        distArrayCell.SetName(destArrayName)


    pdcopy= vtk.vtkPolyData()
    pdcopy.ShallowCopy(pd)

    pdcopy.GetPointData().AddArray(distArrayPoint)
    pdcopy.GetCellData().AddArray(distArrayCell)
    
    return(pdcopy)


def ConvertListPDToPD(pdList):
    allPD=vtk.vtkAppendPolyData()
    for i1,item1 in enumerate(pdList):
        allPD.AddInputData(item1)
    allPD.Update()
    return(allPD.GetOutput())

   


    
        
    
    
    

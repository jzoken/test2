

# usage


# INPUT:
# generate_code_errors directory  calibrationSpecKey 
# direcotry is the folder that contains the images
# calibrationSpecKey (along with the smartphone model name) is part of the key into the calibration table

# 
# directory that contains the image data should also contain a comma delimited file (markers_metadata.csv) which describes
# the expected location of the markers

# OUTPUT
# 
# a file called "markerLocations.txt" is created in the folder that contains the locations of the markers for ground truthing
# debug images photos are also created that show the quality of the codes


# MISCELLANEOUS
# need to add a new entry to the distotionParams[] table for every additional (camera,calibration) combination




from __future__ import print_function

import numpy as np

import cv2

import sys
import os

import scipy.spatial.distance as spsd
import scipy.spatial as spatial


import csv
import re
import PIL.Image
import PIL.ExifTags

import datetime
from time import gmtime, strftime

import time

import pollTireStates
import generate_codesSettings
import tireProcessStateSettings




def FixCode(faultyCode, markerList, numberBadSquaresThreshold):
    for itemCode in markerList:
        unmatched= faultyCode^itemCode
        numUnmatched = bin(unmatched).count("1")
        if (numUnmatched< numberBadSquaresThreshold):
            return(itemCode)
    return(0)
        


def StartWatch(codeWatch):
    global performanceTimeTrackingList
    int(round(time.time() * 1000))
    performanceTimeTrackingList[codeWatch,'StartTime']=int(round(time.time() * 1000))


def EndWatch(codeWatch):
    global performanceTimeTrackingList
    now=int(round(time.time() * 1000))
    elapsed = now - performanceTimeTrackingList[codeWatch,'StartTime']
    performanceTimeTrackingList[codeWatch,'TotalTime']=performanceTimeTrackingList[codeWatch,'TotalTime']+elapsed
    


 


def Distort(undistX,undistY,fx,fy,cx,cy,k1,k2,k3,p1,p2):
    
    x=(undistX-cx)/fx
    y=(undistY-cy)/fy
    
    r2 = x*x* + y*y
    
    xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    
    # Tangential distorsion
    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);
#
    # Back to absolute coordinates.
    xDistort = xDistort * fx + cx;
    yDistort = yDistort * fy + cy;
    
    return(xDistort,yDistort)
    



def persTransformCorners(M, pts):
	

    foo=np.zeros((1,1,2))
    out=np.zeros((1,1,2))
    	
    foo[0,0,0]=pts[0,0]
    foo[0,0,1]=pts[0,1]
    
    outList=[]
         
         
    out=cv2.perspectiveTransform(foo,M)
    outList.append(out)
    print ("\n77777777777777777666666666666666", "\n", foo,"\n", out, "Shape", out.shape, "\n", M)
     
    foo[0,0,0]=pts[1,0]
    foo[0,0,1]=pts[1,1]
    	
     
    out=cv2.perspectiveTransform(foo,M)
    outList.append(out)
    print ("\n77777777777777777666666666666666", "\n", foo,"\n", out, "\n", M)
     
     
    foo[0,0,0]=pts[2,0]
    foo[0,0,1]=pts[2,1]
    	
     
    out=cv2.perspectiveTransform(foo,M)
    outList.append(out)
    print ("\n77777777777777777666666666666666", "\n", foo,"\n", out, "\n", M)
     
     
    foo[0,0,0]=pts[3,0]
    foo[0,0,1]=pts[3,1]
    	
     
    out=cv2.perspectiveTransform(foo,M)
    outList.append(out)
    print ("\n77777777777777777666666666666666", "\n", foo,"\n", out, "\n", M)
     
     
    print ("\n333333333333 Warping Matrix\n", M, "\n")
    print (M)
    print ("\n77773333333333333333333333330000000000", outList)
    
   
   
    
    
    return(outList)





def cv2DebugWrite(fileName,image):
    if (debugFlag):
        cv2.imwrite(fileName,image)



def computeDest(tl, tr, br, bl):
# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
    	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")

    return((dst,maxHeight, maxWidth))
 


def rectify(h):
#        print "\n$$$$$$$$$$ rectify \n", h
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
#        print "\nAdd hnew\n",add, "\n",hnew[0],hnew[2]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
        #print "\n$$$$$$$$$$ rectify \n", h
#        print "\nDiff hnew\n",diff, "\n",hnew[1],hnew[3]
#        print "\nResult \n",hnew
  
        return hnew

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    print ("\n a1,a2, b1,b2", a1,a2, b1,b2, "\n")
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


def findSide(pt0,pt1,cntitem):
    #print "\n findside", pt0,pt1,"\n ", cntitem,"\n"
    for i in range (cntitem.shape[0]):

        xval=cntitem[i][0]
        yval=cntitem[i][1]
        #print "\n xval yval ", xval," ",yval, " ", pt0[0], " ", pt0[1],"\n"
        if ( (xval==pt0[0] )  and (yval==pt0[1]) ):
            index0=i
            break
    
    for i in range (cntitem.shape[0]):
        xval=cntitem[i][0]
        yval=cntitem[i][1]
        if ( (xval==pt1[0] )  and (yval==pt1[1]) ):
            index1=i
            break
    
#    print "findeside end",index0," ",index1
    return(index0,index1)




def getIntersection(line1, line2):
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < sys.float_info.epsilon:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)
 
 
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
    

#

def DetermineIfAtCorner(clockwiseFlag,cx,cy,pt0x,pt1x,pt0y,pt1y):
     if (clockwiseFlag):
         if  (  ( (cx>= pt0x) & ((cx<=pt1x)) ) & 
                ( (cy>= pt0y) & ((cy<=pt1y)) ) ):
                    return(True)
     else:
         if  (  ( (cx>= pt1x) & ((cx<=pt0x)) ) & 
                ( (cy>= pt1y) & ((cy<=pt0y)) ) ):        
                    return(True)
         
     return(False)


     
def clockWiseOrientation(cntFlatten):
    #cntFlatten= cnt.flatten()
    tally=0
    for i in range(len(cntFlatten)-1):
        pt0x=cntFlatten[i,0]
        pt1x=cntFlatten[i+1,0]
        pt0y=cntFlatten[i,1]
        pt1y=cntFlatten[i+1,1]
        tally = tally+ (pt1x-pt0x)*(pt1y+pt0y)
    
    # connect last point to first
    pt0x=cntFlatten[len(cntFlatten)-1,0]
    pt1x=cntFlatten[0,0]
    pt0y=cntFlatten[len(cntFlatten)-1,1]
    pt1y=cntFlatten[0,1]
    tally = tally+ (pt1x-pt0x)*(pt1y+pt0y)
        
    if (tally>0) :
        return(True)
    else:
        return(False)
        


def FindFirstCorner(cntFlatten,clockWiseFlag,corners):

    for i in range(len(cntFlatten)-1):
        pt0x=cntFlatten[i,0]
        pt1x=cntFlatten[i+1,0]
        pt0y=cntFlatten[i,1]
        pt1y=cntFlatten[i+1,1]
   
        if DetermineIfAtCorner(clockWiseFlag,corners[0,0],corners[0,1],pt0x,pt1x,pt0y,pt1y):
            # top left
            return(0,i)
        else:
            if DetermineIfAtCorner(clockWiseFlag,corners[0,0],corners[0,1],pt0x,pt1x,pt1y,pt0y):
                return(1,i)
            else:
                if DetermineIfAtCorner(clockWiseFlag,corners[0,0],corners[0,1],pt1x,pt0x,pt0y,pt1y):
                    return(2,i)
                else:
                    if DetermineIfAtCorner(clockWiseFlag,corners[0,0],corners[0,1],pt1x,pt0x,pt1y,pt0y):
                        return(3,i)
                        
                        
def FindFirstCorner1(cntFlatten,clockWiseFlag,corners):
    foundCorner=[False,False,False,False]
    foundIndex=[0,0,0,0]
    for i in range(len(cntFlatten)-1):
        pt0x=cntFlatten[i,0]
        pt1x=cntFlatten[i+1,0]
        pt0y=cntFlatten[i,1]
        pt1y=cntFlatten[i+1,1]
   
        if (not foundCorner[0]):
            if DetermineIfAtCorner(clockWiseFlag,corners[0,0],corners[0,1],pt0x,pt1x,pt0y,pt1y):
                foundIndex[0]=i
                foundCorner[0]=True        
        else:
            if (not foundCorner[1]):
                if DetermineIfAtCorner(clockWiseFlag,corners[1,0],corners[1,1],pt0x,pt1x,pt1y,pt0y):
                    foundIndex[1]=i
                    foundCorner[1]=True  
            else:
                if (not foundCorner[2]) :
                    if DetermineIfAtCorner(clockWiseFlag,corners[2,0],corners[2,1],pt1x,pt0x,pt0y,pt1y):
                        foundIndex[2]=i
                        foundCorner[2]=True  
                        foundIndex[3]=len(cntFlatten)-1
                        foundCorner[2]=True  
        
    return(foundIndex)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.rad2deg(np.arccos(np.dot(v1_u, v2_u)))
#    if np.isnan(angle):
#        if (v1_u == v2_u).all():
#            return 0.0
#        else:
#            return np.pi
    return angle                    
#                  
def FindCorners(cntFlatten):
    foundCorner=[False,False,False,False]
    foundIndex=[0,0,0,0]
    for i in range(len(cntFlatten)-1):
        pt0x=cntFlatten[i,0]
        pt1x=cntFlatten[i+1,0]
        pt0y=cntFlatten[i,1]
        pt1y=cntFlatten[i+1,1]
        #angleB= angle_between((pt0x,pt0y), ( pt1x,pt1y ))
        diff1=(pt0x-pt1x)
        diff2=(pt0y-pt1y)
        dist=np.sqrt( (diff1*diff1) + (diff2*diff2)  )
        theta=np.arcsin(diff1/dist)
#        hyp=( (pt0x*pt0x) + (pt0y*pt0y) )
        print ("\n i angle",i,np.rad2deg(theta), " ",pt0x,pt0y," ",pt1x,pt1y)
 
        
    return(foundIndex)
                               
            
            
        
    
    
 

def boxFormContourBox3(cnt,boxID, codeBoxArray1,maxNumberOfPtsPerSide):
   
    corners = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
 
    
    corners=rectify(corners)
#    print "\n555555555 Corners 555555555\n", corners,corners[1,1]
    
    
    
    cntFlatten= cnt.flatten()
    

    
    shp = cntFlatten.shape
    cntFlatten=cntFlatten.reshape( (  (shp[0]/2),2  ) )
#    clockWiseFlag=clockWiseOrientation(cntFlatten)
#    
#    clockWiseFlag=True
#    
#    cornerIndices=FindCorners(cntFlatten)
    
    A=cntFlatten
    distance,index = spatial.KDTree(A).query(corners)
    
    



    #sortedIndices=np.argsort(index)
    minItem=np.min(index)
    maxItem=np.max(index)
    
    
    for i in range(4):
        index0=index[i]
        index1=index[(i+1)%4]
        
 
            
        if (   ( ( index[i] ==minItem) & (  index[((i+1)%4)]   ==maxItem) )  | ( (  index[ ((i+1)%4) ] ==minItem) & ( index[ i ] ==maxItem) )   ):
        #if (   ( ( i ==minIndex) & (  ((i+1)%4)   ==maxIndex) )  | ( (   ((i+1)%4)  ==minIndex) & (  i  ==maxIndex) )   ):
            iseq1=np.arange(max(index0,index1),cnt.shape[0]-1,1)
            if (minItem>0):
                iseq2=np.arange(0,min(index0,index1),1)
                iseq3=np.concatenate((iseq1,iseq2))
            else:
                iseq3=iseq1
            ptsonline=cnt[iseq3]
        else:   
            ptsonline=cnt[min(index0,index1):max(index0,index1)]
          
         
           
        
        ptsonlff=ptsonline.flatten()
        ptsonlrs=ptsonlff.reshape(    len(ptsonlff)/2  ,2)
#        print "\n ########## pts on line shape ", ptsonlrs.shape
        
        ptsInInterval=np.zeros( (maxNumberOfPtsPerSide,2))
        ptsInInterval[0:len(ptsonlff)/2]=ptsonlrs
        

             
        codeBoxArray1[boxID,i]=ptsInInterval
    
#    if (corners[1][1]>corners[1][1]):
#    
#    topSeq=np.where( ((cntFlatten[:,0]>corners[0][0]) &  (cntFlatten[:,1]>corners[0][1])  &  (cntFlatten[:,0]<corners[1][0]) ) &  (cntFlatten[:,1]>corners[1][1]) )
#        
    
    
    return

def graphLine1(img,vx,vy,x,y,clr,thckness):

        
    
    xMax=img.shape[1]
    yMax=img.shape[0]
    
    result=[]
    
    nY0= -y/vy
    xY0 = x + nY0*vx
    result.append( (xY0,0) )
    
    nyMax= (yMax-y)/vy
    xyMax = x + nyMax*vx
    result.append((xyMax,yMax))
    
    nX0= -x/vx
    yX0 = y + nX0*vy
    result.append((0,yX0))
    
    nxMax= (xMax-x)/vx
    yxMax = y + nxMax*vy
    result.append((xMax, yxMax))
    
    resultList = [item for item in result if (item[0]>=0) & (item[1]>=0) & (item[0]<=xMax) & (item[1]<=yMax)    ]
    
    resultP1=( (int(resultList[0][0]), int(resultList[0][1]) ) )
    resultP2=( (int(resultList[1][0]), int(resultList[1][1]) ) )
    resultP1f=( ((resultList[0][0]), (resultList[0][1]) ) )
    resultP2f=( ((resultList[1][0]), (resultList[1][1]) ) )

    
    
      
    cv2.line(img,resultP1,resultP2 ,clr,2)
    
    return(  resultP1f, resultP2f)
   



def graphLine1Old(img,vx,vy,x,y,clr,thckness,pointOnLine):

    
    maxX=img.shape[1]
    minX=0
    maxY=img.shape[0]
    minY=0
    
#    print "\n#################################  graphline 1"
#    print "\nvx,vy,x,y,clr,thckness,pointOnLine", vx,vy,x,y,clr,thckness
#    print "\nminX,maxX,minY,maxY", minX,maxX,minY,maxY
    my = vy/vx
    
#    print "\n my is ", my
    
    n=(maxY-y)/my
    xMaxY = x+n
    #print "\n n xMaxY", n,xMaxY
    n=(minY-y)/my
    xMinY = x+n
    #print "\n n xMinY", n,xMinY
    
    mx = vx/vy
    
    #print "\n mx is ", mx
    
    n=(maxX-x)/mx
    yMaxX = y+n
    #print "\n n yMaxX", n,yMaxX
    n=(minX-x)/mx
    yMinX = y+n
    #print "\n n yMinX", n,yMinX
    
     
    
    if ( (xMinY*np.sign(vx)>minX) and (xMaxY*np.sign(vy)<maxX)):
        resultP1=( (int(xMinY), int(minY) ) )
        resultP2=( (int(xMaxY), int(maxY))  )
        resultP1f=( ((xMinY), (minY) ) )
        resultP2f=( ((xMaxY), (maxY))  )
        
        
    else:
        resultP1=( (int(minX), int(yMinX) ) )
        resultP2=( (int(maxX), int(yMaxX))  )
        resultP1f=( ((minX), (yMinX) ) )
        resultP2f=( ((maxX), (yMaxX))  )
        

        
    #print "\n Result is ", resultP1, resultP2
   
    cv2.line(img,resultP1,resultP2 ,clr,2)
    
    return(  resultP1f, resultP2f)


def AssociateBoxesWithRowsColumns(img,cbal):
#    
#    leftMostPointsFromBox=codeBoxArray[rowNumber,columnNumber,3].flatten() [np.flatnonzero(codeBoxArray[rowNumber,columnNumber,3])].reshape(-1,2)
#

    cntsYIntercepts=[]
    for cntItem in cbal:
        #topMostPoints=cntItem[0]   
        topMostPoints=cntItem[0].flatten() [np.flatnonzero(cntItem[0]      )].reshape(-1,2) 
        
        #leftMostPointsFromLeftBoxes=codeBoxArray[:,0,3].flatten() [np.flatnonzero(codeBoxArray[:,0,3])].reshape(-1,2)
        (vxLMB,vyLMB,xLMB,yLMB) = cv2.fitLine(topMostPoints,cv2.DIST_FAIR,0,0.01,0.01).flatten()
        yInterceptInformation=graphLine1(img,vxLMB,vyLMB,xLMB,yLMB,(0,255,0),3)
        #delme=graphLine2(img,vxLMB,vyLMB,xLMB,yLMB,(0,255,0),3,topMostPoints)
        cntsYIntercepts.append( ((yInterceptInformation[0][1]),cntItem) )
        
    cv2DebugWrite('c:\\temp\\associmage.png',img)
    sorted_cntsYIntercepts = sorted(cntsYIntercepts, key=lambda tup: tup[0]) 

    
    yDistanceBetweenContiguousCodes=999
   
    # presumes at least one rows where 2 succ codes are correctly identified
    for i in range(len(sorted_cntsYIntercepts)-1):
        y1=sorted_cntsYIntercepts[i][0]
        y2=sorted_cntsYIntercepts[i+1][0]
        gap= abs(y2-y1)
        if (gap>maxYInterceptGapForRow.maxYInterceptGapForRow) & (gap<yDistanceBetweenContiguousCodes):
            yDistanceBetweenContiguousCodes=gap
    
    start=sorted_cntsYIntercepts[0][0]       
    numRows=int(abs(sorted_cntsYIntercepts[len(sorted_cntsYIntercepts)-1][0]-start+0.5*generate_codesSettings.yDistanceBetweenContiguousCodes)/yDistanceBetweenContiguousCodes)+1
    codeBoxArray1=np.zeros( (numRows,2,4,500, 2) ,dtype = np.int32)
    
    
    start=sorted_cntsYIntercepts[0][0]
    for i in range(len(sorted_cntsYIntercepts)):
        rowNumber=abs(sorted_cntsYIntercepts[i][0]-start+0.5*yDistanceBetweenContiguousCodes)/yDistanceBetweenContiguousCodes
#        print "\n row #  value", int(rowNumber), " ", sorted_cntsYIntercepts[i][0], " ", (start+0.5*yDistanceBetweenContiguousCodes)
        if sorted_cntsYIntercepts[i][1][1,0,0] > 1300:
            codeBoxArray1[int(rowNumber),1]=sorted_cntsYIntercepts[i][1]
        else:
            codeBoxArray1[int(rowNumber),0]=sorted_cntsYIntercepts[i][1]
    
      
    

    return ( codeBoxArray1)
    
   
#    scntsRightSorted = sorted(cntsRight, key = lambda x: x[0,0,1],reverse=False)
#    scntsLeftSorted = sorted(cntsLeft, key = lambda x: x[0,0,1],reverse=False)
    

def decodeBox(img,lengthOfInnerSquare,offsetBorder):
    # convert image to grayscale and threshold to b/w
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   

    
    sideLengthInPixels =im_bw.shape[0]
    sideLengthInMM = (2 * offsetBorder) + (5* lengthOfInnerSquare)
    offsetBorderInPixels = sideLengthInPixels * ( offsetBorder/ sideLengthInMM)
    sideOfCodeInPixels= sideLengthInPixels-2*offsetBorderInPixels
    sideOfSubCodeInPixels = sideOfCodeInPixels/5
#    
    codeTally=0
    cnt=0
    for colNum in range(5):
        for rowNum in range(5):
            i1=int(offsetBorderInPixels+rowNum*sideOfSubCodeInPixels)
            i2=int(offsetBorderInPixels+(rowNum+1)*sideOfSubCodeInPixels)
            i3=int(offsetBorderInPixels+(colNum)*sideOfSubCodeInPixels)
            i4=int(offsetBorderInPixels+(colNum+1)*sideOfSubCodeInPixels)
            #smallcode=im_bw[i1:i2,i3:i4]
            smallcode=im_bw[i3:i4,i1:i2]
            avg=np.average(smallcode)
            if (avg>128) or ( (rowNum==2) and (colNum==2)):
                cnt=cnt+1
                codeTally=codeTally+np.power(2,(rowNum+colNum*5))
            #codeTally=codeTally+
            

    return (codeTally)
 
       
       
def checkCodeNearExpectedHorizontalPosition(x,pixelsPerMM,horizontalCenterOfImageInPixels,horizontalDistanceBetweenCodeCentersInMM):
    thresholdPixelDistanceFudgeFactor = 300
    horizontalDistanceCodeToCenterInPixels=(horizontalDistanceBetweenCodeCentersInMM* pixelsPerMM)/2
    lowXCodePos =(horizontalCenterOfImageInPixels - horizontalDistanceCodeToCenterInPixels)
    highXCodePos=(horizontalCenterOfImageInPixels + horizontalDistanceCodeToCenterInPixels)
    if ((abs(x -lowXCodePos  )<thresholdPixelDistanceFudgeFactor) or (abs(x - highXCodePos)<thresholdPixelDistanceFudgeFactor)):
        return(True)
    else:
        return(False)
       
        
     
    
    
    

        
def processPhoto(codeDir,idir,fx,fy,cx,cy,k1,k2,k3,tangd1,tangd2,lengthOfInnerSquare, offsetBorder,horizontalDistanceBetweenCodeCenters, markerDictionary,  markerList, approxCodeHeightInPixels):    

    StartWatch('BeforeContourProcessing') 
    
    filename = codeDir+"\\"+idir
    p1=filename.find("IMG")
    p2=filename.find(".")
    imageNumber=filename[p1+len("IMG")-1:p2]
    
    print ("\n Image number is ", imageNumber)
    
    print ("\n file is ", filename,"\n")
    

    StartWatch('ReadImage')
   
    image = cv2.imread(filename)
    
    
    EndWatch('ReadImage')
    
    

    approxCodeHeightInPixels=  ( (5 * lengthOfInnerSquare) + (2* offsetBorder) ) * generate_codesSettings.pixelsPerMM
    
    
    approxCodeContourLength=approxCodeHeightInPixels*4
    approxCodeContourArea=approxCodeHeightInPixels*approxCodeHeightInPixels
  
    
    
    K = np.array([[fx,0,cx], [0, fy, cy], [0, 0, 1]])
    #K = np.array([[fx,0,0], [0, fy, 0], [cx, cy, 1]])
    d = np.array([k1,k2, 0, 0,k3]) # just use first two terms (no translation)
    dist=np.array([k1,k2, 0, 0,k3])
    #d=0
    
    
    h, w = image.shape[:2]
    imgHeight,imgWidth=image.shape[:2]
    horizontalCenterOfImage=w/2
    
     
    StartWatch('UndistortProcessing') 

    # undistort
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1,(w,h)) 
    xroi,yroi,wroi,hroi = roi
    #undistimg_w=wroi
    #undistimg_h=hroi
    
    
    undistimgBeforeCrop=np.zeros( (hroi,wroi,3) ,dtype = np.float32)
    undistimgBeforeCrop = cv2.undistort(image, K, d, None, newcameramtx)
    # now crop the image
    undistimg=undistimgBeforeCrop[yroi:yroi+hroi, xroi:xroi+wroi]
    
    # undistort
    #undistimgBeforeCrop1=np.zeros( (hroi,wroi,3) ,dtype = np.float32)
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wroi,hroi),5)
    undistimgBeforeCrop1 = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)
    undistimg1=undistimgBeforeCrop1[yroi:yroi+hroi, xroi:xroi+wroi]
    
    
    undistimg=undistimgBeforeCrop1
    
    EndWatch('UndistortProcessing') 
    
   
    
    StartWatch('BWThresholdProcessing') 
    
    
    cv2DebugWrite('c:\\temp\\image.png',image)
    cv2DebugWrite('c:\\temp\\undistimage.png',undistimg)
    cv2DebugWrite('c:\\temp\\undistimage1.png',undistimg1)
    cv2DebugWrite('c:\\temp\\undistimageBeforeCrop.png',undistimgBeforeCrop)
    cv2DebugWrite('c:\\temp\\undistimageBeforeCrop1.png',undistimgBeforeCrop1)
    

    imsdup=np.zeros_like(undistimg)
    #imsdup[:] = undistimg
    
    imsdup1 = np.empty_like (undistimg)
    imsdup1[:] = undistimg
    

    
    gray1=cv2.cvtColor(undistimg,cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(gray,127,255,1)
    ret,thresh = cv2.threshold(gray1,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    

    
    
    # rough approach to ignoring codes in distorted area
  
    widthQ=int(generate_codesSettings.distortionAreaRemoveFudgeFactor*w)
    heightQ=int(generate_codesSettings.distortionAreaRemoveFudgeFactor*h)
    threshdup=np.zeros_like(thresh)
    threshdup[heightQ:h-heightQ, widthQ:w-widthQ]=thresh[heightQ:h-heightQ, widthQ:w-widthQ]
    thresh=threshdup
    

    
    
    
    cv2DebugWrite('c:\\temp\\thresh.png',thresh)
    
    EndWatch('BWThresholdProcessing') 

    
    EndWatch('BeforeContourProcessing') 
    StartWatch('ContourProcessing') 
    


    #_,contours,h= cv2.findContours(thresh, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    _,contours,h= cv2.findContours(thresh, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imsdup, contours, -1, (255, 0, 0), 3)
    cv2DebugWrite("c:\\temp\\rawcontours.png",imsdup)
    
    
    scnts4=[cntitem1 for cntitem1 in contours if len(cv2.approxPolyDP(cntitem1,0.01*cv2.arcLength(cntitem1,True),True))==4] 
    
     
    
    cv2.drawContours(imsdup, scnts4, -1, (255, 0, 0), 3)
    cv2DebugWrite("c:\\temp\\cont4sides.png",imsdup)
    



    scnts4WithLength=[(cntitem1,   cv2.arcLength(cntitem1,True)  ) for cntitem1 in contours if ( (len(cv2.approxPolyDP(cntitem1,0.01*cv2.arcLength(cntitem1,True),True))==4 ) and (cv2.contourArea(cntitem1)>1300) )] 

    scnts4WithLengthPossibleCode=[ item for item in scnts4WithLength if (item[1]>approxCodeContourLength*0.80) & (item[1]<approxCodeContourLength*1.2) ]
    
    contoursCodesOnly= [cntitem[0] for cntitem in scnts4WithLengthPossibleCode  ]
    fudgeFactorAreaThresholdDiff=8000
    contoursFilteredByArea=  [ cntitem for cntitem in contoursCodesOnly if abs((cv2.contourArea(cntitem)-approxCodeContourArea))  <fudgeFactorAreaThresholdDiff                       ] 
    
          
    



    contoursCodesOnlyFBA= [cntitem for cntitem in contoursFilteredByArea  ]
    contoursPlusMoments=  [   ( cntitem1, cv2.moments(cntitem1) )     for cntitem1 in contoursCodesOnlyFBA ] 
    #contoursPlusCenters=  [    (  int(M['m10']/M['m00']) , int(M['m01']/M['m00']) )    for M  in contoursPlusMoments ] 
    #contoursPlusCenters=  [   ( cm[0], (  int(cm[1]['m10']/cm[1]['m00']),  int(cm[1]['m01']/cm[1]['m00'])    )   )    for cm  in contoursPlusMoments ] 
    contoursPlusCenters=  [   ( cm[0], (  int(cm[1]['m10']/cm[1]['m00']),  int(cm[1]['m01']/cm[1]['m00'])    )   )    for cm  in contoursPlusMoments ] 
    #contoursPlusCenters=  [   ( cm[0], (  int(cm[1]['m10']/cm[1]['m00']),  int(cm[1]['m01']/cm[1]['m00'])    )   )    for cm  in contoursPlusMoments ] 
    print ("\n length of contoursPlusCenters +++ ********\n", len(contoursPlusCenters))
   
    contoursNearExpectedPosition = [ cm[0] for cm in contoursPlusCenters if checkCodeNearExpectedHorizontalPosition(cm[1][0],generate_codesSettings.pixelsPerMM,horizontalCenterOfImage,horizontalDistanceBetweenCodeCenters)   ]
    
   # print ("\n length of contours ********\n", len(contoursNearExpectedPosition))
    #hack
    #contoursNearExpectedPosition=contoursCodesOnly
    print ("\n length of scnts4WithLength ********\n", len(scnts4WithLength))
    print ("\n length of scnts4WithLengthPossibleCode ********\n", len(scnts4WithLengthPossibleCode))
    print ("\n length of contoursCodesOnly ********\n", len(contoursCodesOnly))
    print ("\n length of contoursFilteredByArea ********\n", len(contoursFilteredByArea))
    print ("\n length of contoursPlusMoments ********\n", len(contoursPlusMoments))
    print ("\n length of contoursPlusCenters ********\n", len(contoursPlusCenters))
  
    
    print ("\n length of contoursNearExpectedPosition ********\n", len(contoursNearExpectedPosition))
    
    
    imsdupAll=np.zeros_like(undistimg)
    #cv2.drawContours(imsdupAll, contoursNearExpectedPosition, 3, (255, 0, 0), 3)
    cv2DebugWrite("c:\\temp\\contimgAll.png",imsdup)
    
    for i in range(len(contoursNearExpectedPosition)):
        imsdupAll=np.zeros_like(undistimg)
        cv2.drawContours(imsdupAll, contoursNearExpectedPosition[i], -1, (255, 0, 0), 3)
        filenamec="c:\\temp\\contnum_"+str(i)+".jpg"
        cv2DebugWrite(filenamec,imsdupAll)
        

    
    numberOfBoxes=len(contoursNearExpectedPosition)
    
    
    codeBoxArray1L=np.zeros(  (numberOfBoxes,4,generate_codesSettings.maxNumberOfPtsPerSide, 2) ,dtype = np.int32)

    codeBoxCorners1L=np.zeros( (numberOfBoxes ,4,2) ,dtype = np.float32)
    

    
    for boxID,boxItem  in enumerate(contoursNearExpectedPosition):
        boxFormContourBox3(boxItem,boxID, codeBoxArray1L,generate_codesSettings.maxNumberOfPtsPerSide)     

#
    EndWatch('ContourProcessing') 
    
    #numberOfRows=100
    codeBoxLocalCenter=np.zeros( (numberOfBoxes,2) ,dtype = np.float32)
    codeBoxExactCenter=np.zeros( (numberOfBoxes,2) ,dtype = np.float32)
    codeBoxCode=np.zeros( (numberOfBoxes) ,dtype = np.int32)
  

    
    
    
    totalErrorTally=0
    codeList=[]
    

    
    StartWatch('BoxProcessing') 
 
    
    for boxNumber in range(numberOfBoxes):
            if (np.count_nonzero(codeBoxArray1L[boxNumber])==0):
#                print "\n 77777777777 Continue"
                continue
            print ("\n ******** Box Number ", boxNumber)
            leftMostPointsFromBox=codeBoxArray1L[boxNumber,3].flatten() [np.flatnonzero(codeBoxArray1L[boxNumber,3])].reshape(-1,2)
            rightMostPointsFromBox=codeBoxArray1L[boxNumber,1].flatten() [np.flatnonzero(codeBoxArray1L[boxNumber,1])].reshape(-1,2)
            topMostPointsFromBox=codeBoxArray1L[boxNumber,0].flatten() [np.flatnonzero(codeBoxArray1L[boxNumber,0])].reshape(-1,2)
            bottomMostPointsFromBox=codeBoxArray1L[boxNumber,2].flatten() [np.flatnonzero(codeBoxArray1L[boxNumber,2])].reshape(-1,2)
          
    
          
            [vxLM,vyLM,xLM,yLM] = cv2.fitLine(leftMostPointsFromBox,cv2.DIST_FAIR,0,0.01,0.01).flatten()
            #print "\n $$$$$$$$$$$$$ rightMostPointsFromBox\n",rightMostPointsFromBox
            [vxRM,vyRM,xRM,yRM] = cv2.fitLine(rightMostPointsFromBox,cv2.DIST_FAIR,0,0.01,0.01).flatten()
            [vxTM,vyTM,xTM,yTM] = cv2.fitLine(topMostPointsFromBox,cv2.DIST_FAIR,0,0.01,0.01).flatten()
            [vxBM,vyBM,xBM,yBM] = cv2.fitLine(bottomMostPointsFromBox,cv2.DIST_FAIR,0,0.01,0.01).flatten()
    
        
            topMost=graphLine1(undistimg,vxTM,vyTM,xTM,yTM,(255,255,0),0.5)
            #print "\n *****TM****", rowNumber, columnNumber,vxTM,vyTM,xTM,yTM
            rightMost=graphLine1(undistimg,vxRM,vyRM,xRM,yRM,(255,0,0),0.5)
            #print "\n *****RM****", rowNumber, columnNumber, vxRM,vyRM,xRM,yRM
            bottomMost=graphLine1(undistimg,vxBM,vyBM,xBM,yBM,(0,255,255),0.5)
            #print "\n *****BM****", rowNumber, columnNumber, vxBM,vyBM,xBM,yBM
            leftMost=graphLine1(undistimg,vxLM,vyLM,xLM,yLM,(0,0,255),0.5)
            #print "\n *****LM******", rowNumber, columnNumber,vxLM,vyLM,xLM,yLM
            
            cv2DebugWrite("c:\\temp\\img_state.jpg", undistimg)
            
            topLeft=line_intersection( leftMost, topMost)
            topRight=line_intersection( rightMost, topMost)
            bottomRight=line_intersection( rightMost, bottomMost)
            bottomLeft=line_intersection( leftMost, bottomMost)
            
#            print "\n Corners of Box", topLeft, topRight, bottomRight, bottomLeft
           
            
            codeBoxCorners1L[boxNumber,0]= np.array([ int(topLeft[0]), int(topLeft[1])          ])
            codeBoxCorners1L[boxNumber,1]= np.array([ int(topRight[0]), int(topRight[1])          ])
            codeBoxCorners1L[boxNumber,2]= np.array([ int(bottomRight[0]), int(bottomRight[1])          ])
            codeBoxCorners1L[boxNumber,3]= np.array([ int(bottomLeft[0]), int(bottomLeft[1])          ])
            
           
            
            print ("\n**** ", codeDir, idir, "\n **** Corners",codeBoxCorners1L[boxNumber,0],codeBoxCorners1L[boxNumber,1],codeBoxCorners1L[boxNumber,2],codeBoxCorners1L[boxNumber,3],"\n" )
           
           
          
 
             
          
            destBox1=np.array([  
                [0,0  ],
                [generate_codesSettings.maxSizeCodeBoxInPixels,0 ],
                [generate_codesSettings.maxSizeCodeBoxInPixels,generate_codesSettings.maxSizeCodeBoxInPixels ],
                [0,generate_codesSettings.maxSizeCodeBoxInPixels ],
              
                ],dtype = "float32")
    
    
    
            cv2DebugWrite("c:\\temp\\testme.jpg", imsdup1)                 
            localBoxTransform= cv2.getPerspectiveTransform(  codeBoxCorners1L[boxNumber] ,  destBox1    )
            #localBoxImg=np.zeros( (500,500,3) ,dtype = np.float32)
            filenamebox="c:\\temp\\localBox"+str(boxNumber)+"_.jpg"
            warp = cv2.warpPerspective(imsdup1,localBoxTransform,(generate_codesSettings.maxSizeCodeBoxInPixels,generate_codesSettings.maxSizeCodeBoxInPixels))
            cv2DebugWrite(filenamebox, warp)
            
            code=decodeBox(warp,lengthOfInnerSquare,offsetBorder)
            codeBoxCode[boxNumber]=code
            print ("\n&&&&&&&&&&&&&&& Code is ", code)
            if str(code) not in markerDictionary:            
                code=FixCode(code, markerList, 2)
                if (code==0):
                    codeBoxLocalCenter[boxNumber]=np.array([   0,0       ]) 
                    continue
            
            centerSmall = line_intersection(  (  topLeft, bottomRight     ), ( topRight,  bottomLeft     ) )
            codeBoxLocalCenter[boxNumber]=np.array([centerSmall[0],  centerSmall[1]]) 
            
            codeBoxExactCenter[boxNumber]=np.array((float(markerDictionary[str(code)][0]),float(markerDictionary[str(code)][1])       )).reshape(-1,2)

            codeList.append(code)
            
   

    EndWatch('BoxProcessing') 
            
  
    StartWatch('ErrorMeasureProcessing')     

   
    cbec=codeBoxExactCenter.flatten() [np.flatnonzero(codeBoxExactCenter)].reshape(-1,2)
    cblc=codeBoxLocalCenter.flatten() [np.flatnonzero(codeBoxLocalCenter)].reshape(-1,2)
    numNonZeroCodes=cblc.shape[0]
    
    minNumberCorrectCodesPerPicture=3
    codeErrors=False
    if (numNonZeroCodes<minNumberCorrectCodesPerPicture):
        codeErrors=True
        return ( (codeErrors, 0,0,0,0,0 ) )
    
    #perspTrAllPts,mask=cv2.findHomography(codeBoxLocalCenter.reshape(-1,2), codeBoxExactCenter.reshape(-1,2) )
    perspTrAllPts,mask=cv2.findHomography(cblc,cbec)
    
    if perspTrAllPts is None:
        codeErrors=True
        return ( (codeErrors, 0,0,0,0,0 ) )
        
    #cbmc=np.zeros_like( cbec ,dtype = np.float32)
    
    #codeBoxMappedCenter=cv2.perspectiveTransform( cblc.reshape(1,numberOfRows*2,2),perspTrAllPts)
    cbmc=cv2.perspectiveTransform( cblc.reshape(1,numNonZeroCodes,2),perspTrAllPts)
    
    

    
    imgUndist = np.empty_like (undistimg)
    imgUndist[:] = undistimg
    filenameperscorrected="c:\\temp\\IMGPersCorr.jpg"
    warp = cv2.warpPerspective(imgUndist,perspTrAllPts,(imgWidth,imgHeight))
    cv2DebugWrite(filenameperscorrected, warp)
    
    
    
    distResult=spsd.cdist(cbmc.reshape(-1,2),cbec.reshape(-1,2) )
    
    rmsError=np.sqrt(totalErrorTally)/(numberOfBoxes*2)
    
    perspTrAllPtsInv=np.linalg.inv(perspTrAllPts)
    cbinferredcenters=cv2.perspectiveTransform( cbec.reshape(1,numNonZeroCodes,2),perspTrAllPtsInv)
   
    rvec=np.array([0,0,0])
    tvec=np.array([0,0,0])
    print ("\nRMS ",rmsError)
    
    
    # generate visual of error by drawing circles around the error (based on filename and undistimg)
    showErrorsFileName=filename
    showErrorsFileName=showErrorsFileName.replace("IMG_", "IMAGEcenter_")
    
    showErrors = np.empty_like (undistimg)
    showErrors[:] = undistimg
    
    for numcodes in range(len(codeList)):
    # for each target - draw a circle around the center
        clr=(255,0,255)
        resultP1=( (     int(cbinferredcenters[0,numcodes,0])        , int(cbinferredcenters[0,numcodes,1]) ) )
        print ("\n Pt is ", resultP1)
        
        cv2.circle(showErrors,resultP1,8 ,clr,1)
        #cv2.circle(showErrors,resultP1,8 ,clr,2)
        errorForCode=distResult[numcodes,numcodes]
       
        resultP1TextError= ( (     int(cbinferredcenters[0,numcodes,0]+110)        , int(cbinferredcenters[0,numcodes,1]) ) )
        resultP1TextCode = ( (     int(cbinferredcenters[0,numcodes,0]-310)        , int(cbinferredcenters[0,numcodes,1]) ) )


        #cv2.putText(showErrors,str(errorForCode),resultP1Text,cv2.FONT_HERSHEY_SIMPLEX,clr)
        cv2.putText(showErrors,str(errorForCode), resultP1TextError, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(showErrors,str(codeList[numcodes]), resultP1TextCode, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        
    print ("\n*** SHow error file name is", showErrorsFileName)
    cv2DebugWrite(showErrorsFileName, showErrors)
    
    
    
#    print "\n\n777777777777\n" , topLeft,topRight,bottomLeft,bottomRight
    
    
    
    cv2DebugWrite("c:\\temp\\lineq.png", undistimg)

    EndWatch('ErrorMeasureProcessing')     
    
    return ( (codeErrors, codeList, cbinferredcenters, distResult,mapx,mapy) )

#    
def determineHeightOfCode(distanceToObject,focalLength,realHeightOfObject,imageHeight, sensorHeight):
    
    
    objectHeightInPixels=(focalLength*realHeightOfObject*imageHeight)/(distanceToObject*sensorHeight)
  
    
    return (objectHeightInPixels)


def getExifInformation(codeDir):

    for idir in os.listdir(codeDir):
        print ("\n codeDir , idir\n",codeDir, idir)
        #if idir.endswith("JPG") and idir.startswith("IMG_"):
        if re.search("IMG.+\.[jJJ][pP][gG]",idir):
            print ("\n 1rst\n")
            img = PIL.Image.open("D:\\temp\\TireAuditRoot\\TScansNotInDB\\2016-05-21_14-43-29\\IMG_0000.jpg")
            print ("\n 2nd")
            img = PIL.Image.open(codeDir+ "\\"+ idir)
         
            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in PIL.ExifTags.TAGS
             }
            print ("\n*", idir)
            return(exif)   

################# main starts here ##########

#codeDir is directory that contains photos of the form IMG_DDDDD
#also contains metadata files that contains geometry of markers (markersMetaData.csv)

global debugFlag
debugFlag=False



#global performanceTimeTrackingList

def generateCodes(path ):
    
    codeDir=path
    calibrationInfo=argList[0]
    #distanceToObject=argList[1]

    print ("\nn ************** got here *")
    
    
    global performanceTimeTrackingList

   

    
    performanceTimeTrackingList={}
    performanceTimeTrackingList['EntireProgram','TotalTime']=0
    performanceTimeTrackingList['BeforeContourProcessing','TotalTime']=0
    performanceTimeTrackingList['ContourProcessing','TotalTime']=0
    performanceTimeTrackingList['BoxProcessing','TotalTime']=0
    performanceTimeTrackingList['ErrorMeasureProcessing','TotalTime']=0
    performanceTimeTrackingList['UndistortProcessing','TotalTime']=0
    performanceTimeTrackingList['BWThresholdProcessing','TotalTime']=0
    performanceTimeTrackingList['ReadImage','TotalTime']=0
    performanceTimeTrackingList['NotProcessPhoto','TotalTime']=0
    
    
    StartWatch('EntireProgram') 
        

    
    
    fx=generate_codesSettings.camera_metadeta_distortionParams_fx 
    fy=generate_codesSettings.camera_metadeta_distortionParams_fy
    cx=generate_codesSettings.camera_metadeta_distortionParams_cx
    cy=generate_codesSettings.camera_metadeta_distortionParams_cy
    k1=generate_codesSettings.camera_metadeta_distortionParams_k1
    k2=generate_codesSettings.camera_metadeta_distortionParams_k2
    k3=generate_codesSettings.camera_metadeta_distortionParams_k3 
    
    tangd1=0
    tangd2=0
    
    
    markernamePosList=[]
    #markersMetadata=os.path.dirname(os.path.dirname(codeDir))+"\\markersMetadata.csv"
    
    markerDictionary={}
    markerList=[]
    with open(generate_codesSettings.markersMetadata, 'rt') as f:  
        reader = csv.reader(f, delimiter=',' , quoting=csv.QUOTE_NONE)
        
        #reader = csv.reader(f)
        for row in reader:
            if (len(row)>3):
                print( "hello\n")
                row1=row
                (a,b,c,d,e,f,g,h,i,j,k)=row
                lengthOfInnerSquare=float(a)
                offsetBorder=float(b)
                sizeOfCheckerboardPattern=int(c)
                numberOfCheckerboardRows=int(d)
                numberOfCodes=int(e)
                verticalDistanceBetweenCodeCenters=float(f)
                horizontalDistanceBetweenCodeCenters=float(g)
                heightOfCodesColumn=float(h)
                widthOfCodesRow=float(i)
                pageHeight=float(j)
                pageWidth=float(k) 
            else:
                (codeName, xcenter_mm,ycenter_mm)=row
                markerDictionary[codeName]=(xcenter_mm,ycenter_mm)
                markerList.append(int(codeName))
                
            print ("\nRow is", row," ", len(row),"\n")
    
    
    
    

    
    #distanceToObject=125
    distanceToObject=generate_codesSettings.camera_target_config_distanceToObject
    focalLength=generate_codesSettings.camera_metadeta_focalLength
    #lengthOfInnerSquare=0.8
    #offsetBorder=1.2
    realHeightOfObject = (2 * offsetBorder) + (5* lengthOfInnerSquare)
    imageHeight=generate_codesSettings.camera_metadeta_imageWidth
    sensorHeight=generate_codesSettings.camera_metadeta_sensorHeight
    
    approxCodeHeightInPixels=determineHeightOfCode(distanceToObject,focalLength,realHeightOfObject,imageHeight, sensorHeight)
        
    
    
       
    
    def decodeValue(s):
        try:
            float(s)
            return(float(s))
        except ValueError:
            try: 
                int(s)
                return(int(s))
            except ValueError:
                return(s)
                
    
    codeToIndexMapper={}
    indexToCodeMapper={}
    
    for i,(key,value) in enumerate(markerDictionary.iteritems()):
        codeToIndexMapper[key]=i
        indexToCodeMapper[i]=key
        
    fileIndex=0
    
    photoToIndexMapper={}
    indexToPhotoMapper={}
    for idir in os.listdir(codeDir):
        print (idir)
        if ( idir.endswith("jpg")  or  idir.endswith("JPG") )        and idir.startswith("IMG_"):
            print ("\n got here")
            filenameWithPath=codeDir+idir
            photoToIndexMapper[filenameWithPath]=fileIndex
            indexToPhotoMapper[fileIndex]=filenameWithPath
            fileIndex=fileIndex+1
            

    
    
    
    codeData= codeDir+ "\\" + generate_codesSettings.markerInformationPixelToCode
    

    
    
    totalGoodPhotos=0
    totalBadPhotos=0
    f = open(codeData,'w')

    for idir in os.listdir(codeDir):
        
       
        if ( idir.endswith("jpg")  or  idir.endswith("JPG") )        and idir.startswith("IMG_"):
            filenameWithPath = codeDir+ "\\" + idir
            #(codeList, cbmc, distResult) = processPhoto("c:\\temp\\targets\\live\\targets5bad1.png")
           
            (codeErrors,codeList, cbmc, distResult,mapx,mapy) = processPhoto( codeDir,idir ,fx,fy,cx,cy,k1,k2,k3,tangd1, tangd2,lengthOfInnerSquare,offsetBorder,horizontalDistanceBetweenCodeCenters, markerDictionary,  markerList, approxCodeHeightInPixels)
            if not codeErrors:
            
                StartWatch('NotProcessPhoto') 
                
                avgError= np.mean(np.diagonal(distResult))
                if (avgError>1):
                    totalBadPhotos=totalBadPhotos+1
                    continue
                totalGoodPhotos=totalGoodPhotos+1
                print( "\n ******************** Avg Error ", avgError)
                   
        
              
        
                for numcodes in range(len(codeList)):
                    #distPx,distPy=Distort(cbmc[0,numcodes,0],cbmc[0,numcodes,1],fx,fy,cx,cy,k1,k2,k3,tangd1,tangd2)
                    #distPx,distPy=Distort(cbmc[0,numcodes,1],cbmc[0,numcodes,0],fy,fx,cy,cx,k1,k2,k3,tangd1,tangd2)
                    #xRevertedToOriginal,yRevertedToOriginal= Distort(imgXLocToMapBack,imgYLocToMapBack,fx,fy,cx,cy,k1,k2,k3,tangd1,tangd2)
                   
        
                    
                    print (filenameWithPath, ",", codeList[numcodes],",",int(cbmc[0,numcodes,0]), ",",int(cbmc[0,numcodes,1]), ",",distResult[numcodes,numcodes]),",",30,",",40
                    #print (filenameWithPath, ",", codeList[numcodes],",",int(cbmc[0,numcodes,0]), ",",int(cbmc[0,numcodes,1]), ",",distResult[numcodes,numcodes],file=f)
        
                    xDistorted = mapx[int(cbmc[0,numcodes,1]),int(cbmc[0,numcodes,0])]
                    yDistorted = mapy[int(cbmc[0,numcodes,1]),int(cbmc[0,numcodes,0])]
                    #printList = [filenameWithPath,codeList[numcodes], cbmc[0,numcodes,0], cbmc[0,numcodes,1], distResult[numcodes,numcodes] ,distPx,distPy,xDistorted,yDistorted]
                    printList = [filenameWithPath,codeList[numcodes], cbmc[0,numcodes,0], cbmc[0,numcodes,1], distResult[numcodes,numcodes] ,xDistorted,yDistorted]
    
                    printList = ','.join(map(str, printList)) 
                    print (printList,file=f)
                    
               
                    
                
                EndWatch('NotProcessPhoto') 
                    
    
    
    f.close()
    
    
    
    EndWatch('EntireProgram') 
    
    print ("\n****  Performane Results ***\n", sorted(performanceTimeTrackingList, key=lambda dct: dct[0]) )
    
    
    progPortions = set([l1[0] for l1 in performanceTimeTrackingList])
    keylist=list(progPortions)
    
    for key in keylist:
        tup=(key,'TotalTime')
        print ("%s: %s" % (key, performanceTimeTrackingList[tup]))
        
        



waitingForState=1
nextState=2
sleepTime=2

argList=["checkboard7x11",170]

print("hello")

pollTireStates.pollTireTransactionStatus(generateCodes, tireProcessStateSettings.stateReadyForCodeProcessing, tireProcessStateSettings.stateReadyForMaskProcessing,sleepTime)



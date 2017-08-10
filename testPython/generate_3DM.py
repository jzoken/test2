import csv
import sys
import fnmatch
import os
import PhotoScan
import time
import testpoll
import os.path



import pollTireStates
import generate_codesSettings
import generate_3DSettings
import tireProcessStateSettings




#global fdebug

def processTAMarkers(PhotoScanMarkerFile,markerNamePosList,chunk,cameraDictionary,fdebug):

    #global fdebug
    errorThreshold=9999
    markerInformation=PhotoScanMarkerFile
    print ("\n ********************** Photoscan marker file", PhotoScanMarkerFile)
    with open(markerInformation, 'rt') as f:
        reader = csv.reader(f, delimiter=',' , quoting=csv.QUOTE_NONE)
        for row in reader:
            fdebug.write("\nUnpack\n")
            fdebug.write("\nrow is \n"+ str(row) + " " + str(len(row))+"\n")
    
         
            #imgWithPath,markerName,x,y,err,dc1,dc2=row
            imgWithPath,markerName,dc1,dc2,err,x,y=row
            if (float(err) >= errorThreshold):
                continue
            print ("\nRow is", row,float(err))
            pos= imgWithPath.find("IMG")
            img= imgWithPath[pos:]
            fdebug.write("\n pos img "+ str(pos) +" "+ str(img))
            markerNamePosList.append( (markerName, img,x,y) )

    fdebug.write("\n &&&&&&&&&&& markernameposlist,"+ str(markerNamePosList))  


    def getKey(item):
            return item[0]
    sortedByMarkerNameList=sorted(markerNamePosList, key=getKey,reverse=True)
    
    print (sortedByMarkerNameList)
    
    #add a target
    # associate it with photos
    
    # read a line, photoname, label name, x, y
    # if target does not have an entry in chunk.markers, then create it - marker_index
    # if photo does not have an entry in chunk.photos, then create - photo_index
    # x,y is the pixel coordinate of target center
    # add an entry to chunk.markers[marker_index].projections[photo_index] = (x,y)
    
    #fdebug.write("\n legth of cmarkers after\n",len(chunk.markers))
    
    markerSet=set()
    
    for photoMarkerItem in sortedByMarkerNameList:
        markerSet.add( photoMarkerItem[0])
    
    print ("\n*********** Marker Set", markerSet)
    
    
    markerDictionary={}
    
    for markername in markerSet:
        marker=chunk.addMarker()
        print ("\n marker name is " +  markername)
        fdebug.write("\n marker name  is " +  markername)
        marker.label=markername
        markerDictionary[markername]=marker
    
    
    print ("\n&&&&&&&&&&& Marker Dictionary", markerDictionary)
    
    
    
  
    for row in sortedByMarkerNameList:
            fdebug.write("\n *** row is " +  str(row) + "\n")
            (markername,img,x,y)=row
           
            print ("\n markername is  ", markername,"\n")
            marker=markerDictionary[markername]
    
            #(markername,img,y,x)=row
            print ("\n markername is  ", markername,"\n")
            print ("\n row is  ", row,"\n")
            x=int(float(x))
            y=int(float(y))
            #fdebug.write("\n x , y",x,y,"\n")
            print ("\n img is ", img, len(img),"\n")
            cameraIndex = cameraDictionary[img]
            #fdebug.write("\n cross check cam label is ", chunk.cameras[cameraIndex].label)
            cam=chunk.cameras[cameraIndex]
            marker.projections[cam]=(x,y)
    
    
    #fdebug.write("\nmarker proj\n",marker.projections)
    
    
    #chunk.detectMarkers('12bit',10)
    
    



def generate3D(filePath):
   
    fullPathPhotoDirectoryName=filePath   
    
    # Define: AlignPhotosAccuracy ["high", "medium", "low"]
    
   
 
  
    
    TireAuditDataRoot=os.path.dirname(os.path.dirname(fullPathPhotoDirectoryName))+"\\"
 
    PhotoScanInputGCPFileTireAuditMarkers=generate_3DSettings.photoScanGCPFile
    PhotoScanInputGCPFileAgisoftMarkers=TireAuditDataRoot+"gcp_agisoft.csv"
    PhotoScanInputGCPFileAgisoftMarkers=TireAuditDataRoot+"foo4.txt"
    PhotoScanInputCalibFile=generate_3DSettings.photoScanCalibrationFile

    
    #fdebug.write("\n ********************* TireAuditDataRoot  PhotoScanInputCalibFile  PhotoScanInputGCPFileAgisoftMarkers  PhotoScanInputGCPFileAgisoftMarkers PhotoScanInputCalibFile\n ",TireAuditDataRoot , PhotoScanInputCalibFile , PhotoScanInputGCPFileAgisoftMarkers,  PhotoScanInputGCPFileAgisoftMarkers ,PhotoScanInputCalibFile)
  
        
    PhotoScanPlyFile = fullPathPhotoDirectoryName+"\\"+generate_3DSettings.photoScanPlyName 

    PhotoScanLogFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanLogName
    PhotoScanDebugFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanDebugName
    PhotoScanProjectFile= fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanProjectName
    PhotoScanReprojectionErrorsFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanReprojectionErrorsName
    PhotoScanMarkerFile =  fullPathPhotoDirectoryName+"\\"+generate_codesSettings.markerInformationPixelToCode

    
    
    #fdebug.write("\n*********** Checking for %s \n", PhotoScanProjectFile)
    # if path already has a psz file in exit then go onto nex directory
    if os.path.exists(PhotoScanProjectFile):
        #fdebug.write("\n*********** already proceessed %s \n", PhotoScanProjectFile)
        exit
        
    fdebug=open(PhotoScanDebugFile,'w') 
    flog = open(PhotoScanLogFile,'w') 
    start=time.time()
    
    print ("\n**enum gpu devices", PhotoScan.app.enumGPUDevices() )
    print ("\n**gpu mask ",PhotoScan.app.gpu_mask)
    enumDev=PhotoScan.app.enumGPUDevices() 
    gpu_mask=PhotoScan.app.gpu_mask
    
    fdebug.write("\n**enum gpu devices  "+str(PhotoScan.app.enumGPUDevices()) )
    fdebug.write("\n**gpu mask  "+str(PhotoScan.app.gpu_mask))
 
    #change me
    
    #####pattern = 'IMG_*[02468].JPG'
    pattern = 'IMG_*'
    #print 'Pattern :', pattern
    print ( "abc")
    
    #files = os.listdir('.')
    files = os.listdir(fullPathPhotoDirectoryName)
    
    photoList=[]
    for filename in fnmatch.filter(files,pattern):
        #print ('Filename: %-25s %s' % (name, fnmatch.fnmatch(name, pattern))   )
    	#item = fullPathPhotoDirectoryName+"\\"+filename
    	item = os.path.join(fullPathPhotoDirectoryName,filename)
    	photoList.append(item)
    	
    
    print ("\n777 Photolist is ", photoList)
    
    doc = PhotoScan.Document()
    chunk = doc.addChunk()

    chunk.crs = PhotoScan.CoordinateSystem('LOCAL_CS["Local Coordinates",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
    #successLoad=chunk.loadReference("c:\\temp\\pgcp.csv","csv")
    #successLoad=chunk.loadReference("c:\\temp\\foo1.csv","csv")
    #successLoad=chunk.loadReference("c:\\temp\\dtest.csv","csv")
    #print ("\n ############# Success Load is ", successLoad)
    #chunk.updateTransform()
    
    
    
    
    chunk.label = "New ChunkB"
    
    cameraDictionary={}
    user_calib = PhotoScan.Calibration()
   
    success=user_calib.load(PhotoScanInputCalibFile)
    print ("\n success loading calib file ", success)
    sensor = chunk.addSensor() #creating camera calibration group for the loaded image
    sensor.label = "Calibration Group 1"
    sensor.type = PhotoScan.Sensor.Type.Frame
    #sensor.width = camera.photo.image().width
    
    #sensor.height = camera.photo.image().height
    sensor.width = 5312
    sensor.height = 2988
    sensor.fixed=True   
    sensor.user_calib=user_calib
    #sensor.width = 3264
    #sensor.height = 2448
    #sensor.calibration.cx=1.61822175e+03
    #sensor.calibration.cy=1.26702669e+03
    #sensor.calibration.fx=3.08768833e+03
    #sensor.calibration.fy=3.08786068e+03
    #sensor.calibration.k1=0.23148765
    #sensor.calibration.k2=0.51836559
    #sensor.calibration.k3=-0.48297284
    
#    sensor.calibration.fx = 3.99411182e+03 
#    sensor.calibration.fy = 3.99418122e+03 
#    sensor.calibration.cx = 2.68713926e+03
#    sensor.calibration.cy = 1.51055154e+03
#    sensor.calibration.k1= 0.24503953  
#    sensor.calibration.k2 = -0.80636859 
#    sensor.calibration.k3 = 0.77637451
#    sensor.user_calib.fx = 3.99411182e+03 
#    sensor.user_calib.fy = 3.99418122e+03 
#    sensor.user_calib.cx = 2.68713926e+03
#    sensor.user_calib.cy = 1.51055154e+03
#    sensor.user_calib.k1= 0.24503953  
#    sensor.user_calib.k2 = -0.80636859 
#    sensor.user_calib.k3 = 0.77637451
#    sensor.user_calib = True
    
    
#    sensor.calibration.fx = 4.05913e+03 
#    sensor.calibration.fy = 4.06049e+03 
#    sensor.calibration.cx = 2.68463e+03
#    sensor.calibration.cy =  1.52241e+03
#    sensor.calibration.k1= 0.273712  
#    sensor.calibration.k2 = -1.03971
#    sensor.calibration.k3 =  1.05705
    
    
    chunk.marker_projection_accuracy=0.0002
    
    
    
    for i,filepath in enumerate(photoList):
            fdebug.write("\nxxxxxxxx     filepath" + " " +  filepath)
            camera=chunk.addCamera() 
            camera.sensor = sensor
            pos= filepath.find("IMG")
            img= filepath[pos:]
            #cameraDictionary[img]=i
            cameraDictionary[img]=camera.key
            camera.open(filepath)
            #camera.open("c:\\Projects\\123DCatch\\Tires_25982510_B_Samsung6ChoppedExampleForTesting\\A\\IMG_14.JPG")
            camera.label=img
            #fdebug.write("\n filepath key", filepath, camera.key,camera.photo.image().width,camera.photo.image().height)
            fdebug.write("\n filepath key" +  " " + filepath + " "+ str(camera.key)+ " "+camera.photo.path)
    
    fdebug.write("\n^^^^^^^^^^ Camera Dictionary" + " " + str(cameraDictionary) )
    
    
    fdebug.write ("\n PhotoList &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n" + " " + str(photoList))
    
    
    fdebug.write("hello")


    if (os.path.isfile(PhotoScanMarkerFile)):
        automaticMarkers=False
    else:
        automaticMarkers=True  
        
    fdebug.write ("\n ^^^^^^^^^^^^^^^^^ automatic markers is \n " +  str(automaticMarkers) + "\n")

    
    if (automaticMarkers):
        chunk.detectMarkers(PhotoScan.TargetType.CircularTarget20bit,44)
        print ("\n ^^^^^^^^^^^^^^^^^ loading automatic ground contreol information\n", PhotoScanInputGCPFileAgisoftMarkers,"\n")
        successLoad=chunk.loadReference(PhotoScanInputGCPFileAgisoftMarkers,"csv")

        fdebug.write("\n succes load is" + " "+ str(successLoad))
    else:
    
    
        markerNamePosList=[]
        processTAMarkers(PhotoScanMarkerFile,markerNamePosList,chunk,cameraDictionary,fdebug)
        fdebug.write ("\n ^^^^^^^^^^^^^^^^^ loading ground contreol information\n" + " " + PhotoScanInputGCPFileTireAuditMarkers + "\n")
        #successLoad=chunk.loadReference("c:\\temp\\gcp_25982510.csv","csv")
        successLoad=chunk.loadReference(PhotoScanInputGCPFileTireAuditMarkers)
    
    # load in ground control information 
    # load in masks
    templateName=fullPathPhotoDirectoryName+ "\\" + "M{filename}.JPG"
    #templateName=fullPathPhotoDirectoryName+ "\\" + "M{filename}.JPG"
    #successMask=chunk.importMasks(path=templateName,method='file',tolerance=10)
    	
    #ALIGN PHOTOS
    fdebug.write("---Aligning photos ...")
    fdebug.write("Accuracy: " + generate_3DSettings.matchAccuracy)
    fdebug.write("\nBefore Matching **** \n")
    
    if (generate_3DSettings.matchAccuracy=="Low"):
        accuracyMatch=  PhotoScan.Accuracy.LowAccuracy
    elif (generate_3DSettings.matchAccuracy=="High"):
        accuracyMatch=  PhotoScan.Accuracy.HighAccuracy
    else:
        accuracyMatch= PhotoScan.Accuracy.MediumAccuracy
    
    if (generate_3DSettings.modelAccuracy=="Ultra"):
        accuracyModel= PhotoScan.Quality.UltraQuality
    elif (generate_3DSettings.modelAccuracy=="High"):
        accuracyModel=  PhotoScan.Quality.HighQuality
    elif (generate_3DSettings.modelAccuracy=="Low"):
        accuracyModel= PhotoScan.Quality.LowQuality
    elif (generate_3DSettings.modelAccuracy=="Lowest"):
        accuracyModel= PhotoScan.Quality.LowestQuality
    else:
        accuracyModel= PhotoScan.Quality.MediumQuality  
           
    
    chunk.matchPhotos(accuracy=  accuracyMatch, preselection=PhotoScan.Preselection.GenericPreselection)
    fdebug.write("\nBefore Align **** \n")
    chunk.alignCameras()
    fdebug.write("\nBefore Optimize **** \n")
    chunk.optimizeCameras()
    fdebug.write("\nBefore Build Dense **** \n")
    chunk.buildDenseCloud(quality=accuracyModel) 
    fdebug.write("\nBefore Build Model **** \n")
    chunk.buildModel(surface= PhotoScan.SurfaceType.Arbitrary, interpolation=PhotoScan.Interpolation.EnabledInterpolation , face_count=generate_3DSettings.faceCount)
    fdebug.write("\nBefore Build Texture **** \n")
    
    mapping = PhotoScan.MappingMode.GenericMapping #build texture mapping
    chunk.buildUV(mapping = mapping, count = 1)
    chunk.buildTexture()
    
    #chunk.exportModel(path=PhotoScanPlyFile, format = 'ply', texture_format='jpg')
    chunk.exportModel(PhotoScanPlyFile, format = PhotoScan.ModelFormat.ModelFormatPLY, texture_format=PhotoScan.ImageFormat.ImageFormatJPEG)
    #chunk.exportModel(path_export + "\\model.obj", format = "obj", texture_format='PhotoScan.ImageFormat.ImageFormatJPEG)

    
    PhotoScan.app.update()
    
    #save ground control information a
    chunk.saveReference(PhotoScanReprojectionErrorsFile, PhotoScan.ReferenceFormat.ReferenceFormatCSV              )
#    
#    if not(chunk.saveReference(PhotoScanReprojectionErrorsFile, PhotoScan.ReferenceFormat.ReferenceFormatCSV              )):
#       app.messageBox("Saving GCP info failed!")
#    
    # SAVE PROJECT
    fdebug.write("---Saving project...")
    fdebug.write("File: " + PhotoScanProjectFile)
    doc.save(PhotoScanProjectFile)
 
    
       
    
    doc.chunks.remove(chunk)
    end=time.time()
    fdebug.close()
    flog.write('\nelapsed time ='+str(end-start)) # python will convert \n to os.linesep
    flog.close()
    return


#fp="C:\\Projects\\123DCatch\\TireEagle19560R15_New_KinkosSamsungAForTesting\\A\\"
#fp="C:\\Projects\\123DCatch\\TireEagle19560R15_New_KinkosSamsungAForTesting\\A\\"
#fp="D:\\temp\\tirepythontest\\tobeprocessed\\2015-11-24_16-09-15\\"
#fp="D:\\temp\\tirepythontest\\DeleteMe\\b\\"
#
#argList=["checkboard7x11",170]
###
#
#fp="D:\\temp\\delme1\\lnewclamp\\"
#fp="D:\\temp\\delme2\\b\\"
##fp="D:\\temp\\delme2\\inewclamp130mmHMaskWornEagle\\"
##fp="D:\\temp\\delme2\\inewclamp130mmHMaskNewEagle\\"
#fp="D:\\temp\\delme2\\inewclamp130mmHMaskWornEagleInflatedReal\\"
##fp="D:\\temp\\delme2\\April26Clamp130mmByHandNewEagle\\"
#fp="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanAppA\\"
#fp="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanApp_2016-05-21_11-02-09\\"
#fp="D:\\temp\\delme2\\inewclamp130mmWornEagleInflatedNewTireScanApp_2016-05-21_14-43-28_tw\\"
#fp="D:\\temp\\TireAuditRoot\\TireScans\dup\\"


#fp="D:\\temp\\\inewclamp130mmHMaskNewEagleInflated\\"

#
#ret=ProcessAgisoft(fp, argList) 
  

pollTireStates.pollTireTransactionStatus(generate3D, tireProcessStateSettings.stateReadyFor3DProcessing, tireProcessStateSettings.stateReadyForReportProcessing,5)


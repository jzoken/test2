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



def generate3D(filePath):
   
    fullPathPhotoDirectoryName=filePath   
    
    # Define: AlignPhotosAccuracy ["high", "medium", "low"]
    #change me
    print ("\n**enum gpu devices", PhotoScan.app.enumGPUDevices() )
    print ("\n**gpu mask ",PhotoScan.app.gpu_mask)
 
  
    
    TireAuditDataRoot=os.path.dirname(os.path.dirname(fullPathPhotoDirectoryName))+"\\"
 
#1    PhotoScanInputGCPFileTireAuditMarkers=generate_3DSettings.photoScanGCPFile
#1    PhotoScanInputGCPFileAgisoftMarkers=TireAuditDataRoot+"gcp_agisoft.csv"
#1    PhotoScanInputGCPFileAgisoftMarkers=TireAuditDataRoot+"foo4.txt"
    PhotoScanInputCalibFile=generate_3DSettings.photoScanCalibrationFile

    
    #fdebug.write("\n ********************* TireAuditDataRoot  PhotoScanInputCalibFile  PhotoScanInputGCPFileAgisoftMarkers  PhotoScanInputGCPFileAgisoftMarkers PhotoScanInputCalibFile\n ",TireAuditDataRoot , PhotoScanInputCalibFile , PhotoScanInputGCPFileAgisoftMarkers,  PhotoScanInputGCPFileAgisoftMarkers ,PhotoScanInputCalibFile)
  
        
    PhotoScanPlyFile = fullPathPhotoDirectoryName+"\\"+generate_3DSettings.photoScanPlyName 

    PhotoScanLogFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanLogName
    PhotoScanDebugFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanDebugName
    PhotoScanProjectFile= fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanProjectName
    PhotoScanReprojectionErrorsFile = fullPathPhotoDirectoryName+"\\"+   generate_3DSettings.photoScanReprojectionErrorsName
    #PhotoScanMarkerFile =  fullPathPhotoDirectoryName+"\\"+generate_codesSettings.markerInformationPixelToCode

    
    
    #fdebug.write("\n*********** Checking for %s \n", PhotoScanProjectFile)
    # if path already has a psz file in exit then go onto nex directory
    if os.path.exists(PhotoScanProjectFile):
        #fdebug.write("\n*********** already proceessed %s \n", PhotoScanProjectFile)
        exit
        
    fdebug=open(PhotoScanDebugFile,'w') 
    flog = open(PhotoScanLogFile,'w') 
    start=time.time()
    
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

    
    
    #1 chunk.marker_projection_accuracy=0.0002
    
    
    
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




    templateName=fullPathPhotoDirectoryName+ "\\" + "M{filename}.JPG"
    #templateName=fullPathPhotoDirectoryName+ "\\" + "M{filename}.JPG"
    #successMask=chunk.importMasks(path=templateName,method='file',tolerance=10)
    successMask=chunk.importMasks(path=templateName,source=PhotoScan.MaskSourceFile,operation=PhotoScan.MaskOperationReplacement,tolerance=10,cameras=chunk.cameras)
    	
    print("\n***********************&&&&&&&&&&&&&&&&& successMask ************",successMask)
    
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
           
    
    chunk.matchPhotos(accuracy=  accuracyMatch, preselection=PhotoScan.Preselection.GenericPreselection,filter_mask=True)
    fdebug.write("\nBefore Align **** \n")
    chunk.alignCameras()
    fdebug.write("\nBefore Optimize **** \n")
    chunk.optimizeCameras()
    fdebug.write("\nBefore Build Dense **** \n")
    chunk.buildDenseCloud(quality=accuracyModel) 
    fdebug.write("\nBefore Build Model **** \n")
    #chunk.buildModel(surface= PhotoScan.SurfaceType.Arbitrary, interpolation=PhotoScan.Interpolation.EnabledInterpolation , face_count=generate_3DSettings.faceCount)

    #chunk.buildModel(surface= PhotoScan.SurfaceType.Arbitrary, interpolation=PhotoScan.Interpolation.EnabledInterpolation , face_count=300000)
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




pollTireStates.pollTireTransactionStatus(generate3D, tireProcessStateSettings.stateSmartPhoneOnlyReadyFor3DProcessing, tireProcessStateSettings.stateSmartPhoneOnlyReadyForReportProcessing,5)


import numpy as np
from PIL import Image
#Self declared libraries and functions
from metadata import metadata_and_time
from mat_to_txt import matTotxt
from cam_calib import caliberate_cam


ids = [8902,8937] #List storing image ids
metadatas = [] #list storing the metadata of the images which is in the form of dictionary 
Serial_Time = [] #List storing the date of capture of image in the form of serial time
dem_text_list = []  #List for storing the dem txt file stored in .mat format

#Using the function metadata_and_time, printing as well as storing the metadata and serial time of the images
metadatas, Serial_Time = metadata_and_time(ids,metadatas,Serial_Time)

#Loading and printing the GCP file 
GCP_A = open('Input_Files\gcp8902.txt', 'r').read()
print('\n\nGCPA_OF THE_First_Image = ', GCP_A)


#DEM File 
doc_name = 'Input_Files\dem.mat' #a variable to store the file name which will be the parameter to a function
# function that converts .mat file to txt and print them and returns a list containing the name of txt files
dem_text_list = matTotxt(doc_name) 

#Determine camera parameters for image 8902 given data or known data

focal_length = 30 
sensor_size = np.array([22.0, 14.7])
pil_img = Image.open('Input_Files\IMG_8902.jpg')
sz_8902 = np.array(pil_img.size)
f =[] #focal length in pixcel coordinate 
for i in range(len(sensor_size)):
  f.append(sz_8902[i]*(focal_length/sensor_size[i]))

print('\n\nFocal Length of Image 8902 in Pixcel coordinate = ',f)

#Camera Caliberation 
cam_A = {} #a dictionary to store thr return from the caliberate function which will create a camera object

cam_A = caliberate_cam() #function call
print('Camera_object',cam_A)

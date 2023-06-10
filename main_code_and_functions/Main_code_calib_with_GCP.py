import numpy as np
from PIL import Image
#Self declared libraries and functions
from metadata import metadata_and_time as mt
from mat_to_txt import matTotxt
from gcp_txt_to_matrix import save_txt_to_matrix 
from Camera import focal_in_pixelunits as fp
from Camera import cam_constructor as cam
from Camera import project 
from Camera import optimizeCam


ids = [8902,8937] #List storing image ids
metadatas = [] #list storing the metadata of the images which is in the form of dictionary 
Serial_Time = [] #List storing the date of capture of image in the form of serial time
dem_text_list = []  #List for storing the dem txt file stored in .mat format

#Using the function metadata_and_time, printing as well as storing the metadata and serial time of the images
metadatas, Serial_Time = mt(ids,metadatas,Serial_Time)

#Loading and printing the GCP file and coverting the GCP to matx
GCP_A = open('Input_Files\gcpatest_xyz.txt', 'r').read() #KEEP IN MIND THis is a new GCPA FILE NOT THE ONE USED IN MATLAB CODE
print('\n\nGCPA_OF THE_First_Image = ', GCP_A)

GCP_A_MATRIX = save_txt_to_matrix('Input_Files\gcpatest_xyz.txt')

#DEM File 
doc_name = 'Input_Files\dem.mat' #a variable to store the file name which will be the parameter to a function
# function that converts .mat file to txt and print them and returns a list containing the name of txt files
dem_text_list = matTotxt(doc_name)

#Assigning the known parameters
focal_length = 30 
sensor_size = np.array([22.0, 14.7])
cameralocation = np.array([446722.0, 7396671.0, 770.0])
viewDirct = [200, 0, 0]

#opening the image in pil format and getting its size
pil_img = Image.open('Input_Files\IMG_8902.jpg')
sz_8902 = np.array(pil_img.size)

#calculating pixcel coordinates focal length
f = fp(focal_length,sz_8902,sensor_size)

#Creating a camera object for storing the parameters of Camera A
#Initializing a dictionary
camA = {}

camA = cam(cameralocation,sz_8902,viewDirct,f) #calling a function and creating the object

#Prininting the Camera object
for key, value in camA.items():
    print(key, value)
    print('\n')

# Optimize camera

#PROJECTION FUNCTION IS WORKING FINE



import numpy as np
from PIL import Image

#Self declared libraries and functions
from metadata import metadata_and_time as mt
from gcp_txt_to_matrix import save_txt_to_matrix as mtx
from Camera1 import focal_in_pixelunits as fp
from Camera1 import cam_constructor as cam
from Camera1 import optimize_cam as oc
from Camera1 import project as pj

#Getting metadata of the image or images
paths = ['KR2_2014_11.JPG']

metadatas = [] #list storing the metadata of the images which is in the form of dictionary 
Serial_Time = [] #List storing the date of capture of image in the form of serial time

#Using the function metadata_and_time, printing as well as storing the metadata and serial time of the images
metadatas, Serial_Time = mt(paths,metadatas,Serial_Time)


#Input parameters of the image
focal_length = 30 
sensor_size = np.array([22.0, 14.7])
img_path1 = 'KR2_2014_11.JPG'
pil_img = Image.open(img_path1)#Reading the image in pil format
sz_8902 = np.array(pil_img.size)
cameralocation = np.array([447948.820, 8759457.100, 407.092])
viewDirct = [275.5354, 3.3047059, 8.5451739] 

#Getting the GCPxuz abd GCPuv and storing it in matrices 
gcp = 'KR2_2014.txt'
gcpvalues = []
gcpvalues = mtx(gcp)
xyz = gcpvalues[:,:3]
uv = gcpvalues[:,3:]

#Finding the focal length in pixcel units
f = fp(focal_length,sz_8902,sensor_size)

#Forming a camera object with the camera parameters
camA = {}
camA = cam(cameralocation,sz_8902,viewDirct,f)

print('\n\n\n\n -------CAM OBJECT------ \n\n')
#Printing the camera object
for key, value in camA.items():
    print(key, value)
    print('\n')

#Optimizing  the camera
Optimizedcam = oc(camA,xyz,uv,[1,0,1,0,0,0,0],img_path1,show=True)

#Printing the optimized cam
if Optimizedcam != None:
    print('\n\n\n\n -------OPTIMIZED CAM------ \n\n')
    for key, value in Optimizedcam.items():
        print(key, value)
        print('\n')
    







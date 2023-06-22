
import numpy as np
from PIL import Image
#Self declared libraries and functions
from gcp_txt_to_matrix import save_txt_to_matrix as mtx
from Camera1 import focal_in_pixelunits as fp
from Camera1 import cam_constructor as cam
from Camera1 import optimize_cam as oc
from Camera1 import project as pj



focal_length = 30 
sensor_size = np.array([22.0, 14.7])
img_path1 = 'KR2_2014_1.JPG'
pil_img = Image.open(img_path1)
sz_8902 = np.array(pil_img.size)
cameralocation = np.array([446722.0, 7396671.0, 770.0])
# viewDirct = [4.80926012548059, 0.0576786149965093, 0.149141422649056]
viewDirct = [275.5354, 3.3047059, 8.5451739] #[0,0,0]

gcp = 'KR2_2014.txt'
gcpvalues = []
gcpvalues = mtx(gcp)
xyz = gcpvalues[:,:3]
uv = gcpvalues[:,3:]


f = fp(focal_length,sz_8902,sensor_size)

camA = {}

camA = cam(cameralocation,sz_8902,viewDirct,f)

for key, value in camA.items():
    print(key, value)
    print('\n')

##code for optimization with cheker board images. To run the program fast I updated the R and T coeff with the value
##I got while running the below code as the initial R and T coeff within the cam_constructor function
# images = glob.glob('calib_images\*.JPG')
# cm_mtx,rad,tan,error = caliberate_cam(images)
# # Camparams = caliberate_cam(images)
# # coeff = Camparams['distortion_coeff']
# camA['Rcoeff'] = rad
# camA['Tcoeff'] = tan
# camA['Full Model'][5] = rad
# camA['Full Model'][6] = tan

# print("\n\n -------------Cam after image caliberation ---------\n\n")

# for key, value in camA.items():
#     print(key, value)
#     print('\n')


Optimizedcam = oc(camA,xyz,uv,[0,0,1,0,0,0,0],img_path1,show=True)

#Only view direction and only Rand T are working but VRT is not 

if Optimizedcam != None:
    print('\n\n\n\n -------OPTIMIZED CAM------ \n\n')
    for key, value in Optimizedcam.items():
        print(key, value)
        print('\n')
    
 









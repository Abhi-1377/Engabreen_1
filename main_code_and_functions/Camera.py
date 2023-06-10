#----------function to calculate focal length in pixcel coordinate-------------
def focal_in_pixelunits(focalLength,imgsz,sensor_size): 
    import numpy as np
    #initializing the matrix or list
    f = []
    #calculating the focal length and appending to f
    for i in range(len(sensor_size)):
       f.append(imgsz[i]*(focalLength/sensor_size[i]))

    F = np.array(f) #converting liat to array
    return F




#------------Function to find the principle point of the image--------
def principle_point(imgsz): #argument is a list and returns an array
    import numpy as np

    #initializing the matrix or list
    c = []

    #calculating the focal length and appending to c
    for sz in imgsz:
        c.append((sz+1)/2)
    
    #converting liat to array
    C = np.array(c)
    return C





#---------------function to find the rotation matrix------------

def rotation_mtx(view_dir):  
     import numpy as np
     #finding the sine and cosine of elements in the view_dir
     S = np.sin(view_dir)
     C = np.cos(view_dir)

     # Formulas for the elements in the rotation matrix
     R = np.array([(S[2]*S[1]*C[0]) -(C[2]*S[0]), (S[2]*S[1]*S[0]) +(C[2]*C[0]), S[2]*C[1], (C[2]*S[1]*C[0])+(S[2]*S[0]), (C[2]*S[1]*S[0])-(S[1]*C[0]), C[2]*C[1], C[1]*C[0], C[1]*S[0], -1*S[1]]).reshape(3,3)
     # for the first two rows complete and replace them with the negative of their value
     R[:2, :] = -R[:2, :]

     return R




#----------function to form the camera constructor -----------
def cam_constructor(cam_loc, imgsz, view_direction, f): #cam_loc, imgsz, f are in array type
    import math
    import numpy as np

    #a check for deformed images with length of image size
    if len(imgsz)<2:
        print('The image is maldeformed hense cannot create constructor')
        return
    
    else:

        #intialising a list 
        dir = []
        #converting view direction from degrees to radians
        for direc in view_direction:
            dir.append((direc*math.pi)/180)
        
        view_dir = np.array(dir) #coverting the direction list to 1a 1d array

        #calling a function to calculate the principle point(return type array)
        c = principle_point(imgsz)

        # initializing rotational coeff (k) and translational coeff(p) with zeros
        k = np.zeros((1, 6))
    
        p = np.zeros((1, 2))

        # calling th efunction to calculate the rotation matrix
        R = rotation_mtx(view_dir)

        #creating an array which stores all the camera parameters
        full_model = np.array([cam_loc[0],cam_loc[1],cam_loc[2],imgsz[0],imgsz[1],view_dir[0],view_dir[1],view_dir[2],f[0],f[1],c[0],c[1],k[0][0],k[0][1],k[0][2],k[0][3],k[0][4],k[0][5],p[0][0],p[0][1]])
        
        #returning a dictionary 
        return {'camera location': cam_loc, 'image size': imgsz, 'view direction': view_dir, 'f':f, 'c':c, 'Rcoeff':k, 'Tcoeff': p, 'Rotational matrix':R,'Full Model': full_model}






#----------function to project 3d coordinates to 2d coordinates -----------

def project(cam, xyz):
    import numpy as np

    #if the gcp file is in such a format such that the elements are orriented across the rows then transpose it to have 3 colums
    if xyz.shape[1]> 3:
        xyz = xyz.T
    
    #orient origin to the cam location by subtracting the cam location vector from the gcp
    xyz = xyz - cam['camera location']

    #Taking the dot product with the transpose of rotation matrix
    xyz = np.dot(xyz, cam['Rotational matrix'].T)

    #dividing the first two colums of xyz with the last column
    xy = xyz[:,0:2]/xyz[:,2].reshape(-1,1)

    #check if any element in Rot coeff or tran coeff is zero
    if np.any(cam['Rcoeff'] != 0) or np.any(cam['Tcoeff'] != 0):

        #create a r2 array which stores the sumof the squares of xy array row wise
        r2 = np.sum(xy**2,axis=1)
        #limit the max value of r2 to 4
        r2[r2>4] = 4

        if np.any(cam['Rcoeff'][2:6] != 0):
             a = (1 + cam['Rcoeff'][0] * r2 + cam['Rcoeff'][1] * r2**2 + cam['Rcoeff'][2] * r2**3) / (1 + cam['k'][3] * r2 + cam['k'][4] * r2**2 + cam['k'][5] * r2**3)
        else:
            a = 1 + cam['Rcoeff'][0] * r2 + cam['Rcoeff'][1] * r2**2 + cam['Rcoeff'][2] * r2**3

        #element wise multiplication between the elements of xy
        xty = xy[:,0] * xy[:,1]

        #modify the xy matrix with following condition
        xy = np.column_stack([
            a * xy[:, 0] + 2 * cam['p'][0] * xty + cam['p'][1] * (r2 + 2 * xy[:, 0]**2),
            a * xy[:, 1] + 2 * cam['p'][0] * xty + cam['p'][1] * (r2 + 2 * xy[:, 1]**2)
        ])

    #modify the uv matrix with following condition
    uv = np.column_stack([
        cam['f'][0] * xy[:, 0] + cam['c'][0],
        cam['f'][1] * xy[:, 1] + cam['c'][1]
    ])

    #if any z value in xyz is zero or negative make the corresponding uv value as nan
    uv[xyz[:, 2] <= 0, :] = np.nan

    #Assign all the z values to a variable
    depth = xyz[:, 2] if len(xyz.shape) > 1 else None

    inframe = (depth > 0) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] <= cam['image size'][1]) & (uv[:, 1] <= cam['image size'][0]) if len(xyz.shape) > 1 else None

    return {'uv':uv,'depth': depth, 'inframe':inframe}






 #------------------------Optimize cam function-----------------

def optimizeCam(cam,xyz,uv,freeparams):
    import numpy as np
    
    #check whether any element is nan in xyz or uv along the row to create a boolean array
    nanrows1 = np.array([np.any(np.isnan(row)) for row in xyz])
    nanrows2 = np.array([np.any(np.isnan(row)) for row in uv])

    #from xyz and uv delete those rows where nan values are there 
    xyz = np.delete(xyz, np.where(nanrows1), axis=0)
    uv = np.delete(uv, np.where(nanrows2), axis=0)

    #extract the full model data from the cam dictionary using the key
    fullmodel0 = cam['Full Model']

    #create a boolean array in which True is when the freeparams value is 1
    freeparams = ~((freeparams == 0)).reshape(-1)
    
    paramix = np.where(freeparams)[0]  # Find indices of True elements

    Nfree = len(paramix)  # Number of True elements


    mbest = np.zeros(Nfree)  #create a 1 d array of zeros with Nfree number of elements  


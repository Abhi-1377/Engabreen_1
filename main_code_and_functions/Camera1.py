
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
        
        k = np.array([0,0,0,0,0,0])
        p = np.array([0,0])


        # calling th efunction to calculate the rotation matrix
        R = rotation_mtx(view_dir)

        #creating an array which stores all the camera parameters
        full_model = [cam_loc,imgsz,view_dir,f,c,k,p]

        #returning a dictionary 
        return {'camera location': cam_loc, 'image size': imgsz, 'view direction': view_dir, 'f':f, 'c':c, 'Rcoeff':k, 'Tcoeff': p, 'Rotational matrix':R,'Full Model': full_model}






#----------function to project 3d coordinates to 2d coordinates -----------

def project(cam, xyz):
    import numpy as np


    #if the gcp file is in such a format such that the elements are orriented across the rows then transpose it to have 3 colums
    if xyz.shape[1]> 3:
        xyz = xyz.T
    XYZ = xyz
    #orient origin to the cam location by subtracting the cam location vector from the gcp
    xyz = xyz - cam['camera location']
    R = rotation_mtx(cam['view direction'])

    #Taking the dot product with the transpose of rotation matrix
    xyz = np.dot(xyz, R.T)

    #dividing the first two colums of xyz with the last column
    xy = xyz[:,0:2]/xyz[:,2].reshape(-1,1)

    #check if any element in Rot coeff or tran coeff is zero
    if np.any(cam['Rcoeff'] != 0) or np.any(cam['Tcoeff'] != 0):

        #create a r2 array which stores the sumof the squares of xy array row wise
        r2 = np.sum(xy**2,axis=1)
        
        #limit the max value of r2 to 4
        r2[r2>4] = 4

        if np.any(cam['Rcoeff'][2:6] != 0):
             a = (1 + cam['Rcoeff'][0] * r2 + cam['Rcoeff'][1] * r2**2 + cam['Rcoeff'][2] * r2**3) / (1 + cam['Rcoeff'][3] * r2 + cam['Rcoeff'][4] * r2**2 + cam['Rcoeff'][5] * r2**3)
             
        else:
            a = 1 + cam['Rcoeff'][0] * r2 + cam['Rcoeff'][1] * r2**2 + cam['Rcoeff'][2] * r2**3

        #element wise multiplication between the elements of xy
        xty = xy[:,0] * xy[:,1]
        
        #modify the xy matrix with following condition
        xy = np.column_stack([
            a * xy[:, 0] + 2 * cam['Tcoeff'][0] * xty + cam['Tcoeff'][1] * (r2 + 2 * xy[:, 0]**2),
            a * xy[:, 1] + 2 * cam['Tcoeff'][0] * xty + cam['Tcoeff'][1] * (r2 + 2 * xy[:, 1]**2)
        ])
    
    
    #modify the uv matrix with following condition
    uv = np.column_stack([
        cam['f'][0] * xy[:, 0] + cam['c'][0],
        cam['f'][1] * xy[:, 1] + cam['c'][1]
    ])

    #if any z value in xyz is zero or negative make the corresponding uv value as nan
    uv[XYZ[:, 2] <= 0, :] = np.nan

    #Assign all the z values to a variable
    depth = xyz[:, 2] if len(xyz.shape) > 1 else None

    inframe = (depth > 0) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] <= cam['image size'][1]) & (uv[:, 1] <= cam['image size'][0]) if len(xyz.shape) > 1 else None

    return uv,depth,inframe



# -----------------------Constructing cam with all variables ---------
#A secondary cam construction function if variables like cam_loc,imgsz view_direction, f, c,k,p are already found
#This function just calculate a rotation matrix with the optimized function and create a camera onject with all
# the parameters in the form of a dictionary

def const_cam_sec(cam_loc, imgsz, view_direction, f, c,k,p):

    #a check for deformed images with length of image size
    if len(imgsz)<2:

        print('The image is maldeformed hense cannot create constructor')
        return
    
    else:

        # calling th efunction to calculate the rotation matrix
        R = rotation_mtx(view_direction)
        #print("\n\n rotation matrix = ",R)

        #creating an array which stores all the camera parameters
        full_model = [cam_loc,imgsz,view_direction,f,c,k,p]

        return {'camera location': cam_loc, 'image size': imgsz, 'view direction': view_direction, 'f':f, 'c':c, 'Rcoeff':k, 'Tcoeff': p, 'Rotational matrix':R,'Full Model': full_model}




#---------------------Residual UV------------------------


def ResidualUV(params,stable,paramix,xyz,uv):
    import numpy as np

    #various cases according to the optimising conditions 

    # Case 1 Optimizing all
    if all(x in paramix for x in [0,2,3,4,5,6]):
        
        cam_loc = params[0:3]
        view_direction = params[3:6]
        f = params[6:8]
        c = params[8:10]
        k = params[10:16]
        p = params[16:]
        
        imgsz = stable
        

        #2 optimising f,c,rotation and translation coeff
    elif all(x in paramix for x in [3,4,5,6]):
        f = params[0:2]
        c = params[2:4]
        k = params[4:10]
        p = params[10:]
        
        cam_loc = stable[0:3]
        imgsz = stable[3:5]
        view_direction = stable[5:] 
        
        

        #3 optimizing only view direction and camera location
    elif all(x in paramix for x in [0,2]):
        cam_loc = params[0:3]
        view_direction = params[3:]
        
        
        imgsz = stable[0:2]
        f = stable[2:4]
        c = stable[4:6]
        k = stable[6:12]
        p = params[12:]
        
        

    #4 optimizing only view direction 
    elif 2 in paramix:
        view_direction = params

        cam_loc = stable[0:3]
        imgsz = stable[3:5]
        f = stable[5:7]
        c = stable[7:9]
        k = stable[9:15]
        p = stable[15:]
        
    
    else: 
        return 0

    
    #With the internal and external parameetrs from above form a camera object which will be the 
    # input for the project fuction
    cam = const_cam_sec(cam_loc, imgsz, view_direction, f, c,k,p)
    

    # calling the project function for the camera object described above 
    uv_projected,depth,inframe  = project(cam,xyz)
  

    #a list to store the squared difference between the project and actual 2d coordinates
    residual = []

    for i in range(len(uv_projected)):
        residual.append(np.sqrt((uv_projected[i][0]-uv[i][0])**2 + 
                                (uv_projected[i][1]-uv[i][1])**2))
        

    #converting list to array 
    residual = np.array(residual)

    return residual
        

 #------------------------Optimize cam function-----------------


def optimize_cam(cam,xyz,uv,freeparams,img_path,optmethod='trf',show = False):
    import numpy as np
    from scipy import optimize
    from PIL import Image
    from plot_residual import plotResiduals
    from plot_residual import readImg

    #checking whether the matrix xyz or uv contains any NAN values
    nanrows1 = np.array([np.any(np.isnan(row)) for row in xyz])
    nanrows2 = np.array([np.any(np.isnan(row)) for row in uv])


    #from xyz and uv delete those rows where nan values are there 
    xyz = np.delete(xyz, np.where(nanrows1), axis=0)
    uv = np.delete(uv, np.where(nanrows2), axis=0)

    GCPxyz_proj0,depth,inframe = project(cam, xyz)

    

    #extract the full model data from the cam dictionary using the key
    fullmodel0 = cam['Full Model']

    #create a boolean array in which True is when the freeparams value is 1
    freeparams = [True if value == 1 else False for value in freeparams]
 
    paramix = np.where(freeparams)[0]  # Find indices of True elements

    print("\n\n freeparams = ", freeparams)
    print("\n\n paramix = ", paramix)

    #defining two lists to store the variable and stable parameter accroding to the optimization criteria
    params= []
    stable = []

    for i in paramix:
        params.append(fullmodel0[i]) # appending those values in fullmodel0 whose indices are there in paramix

    if len(params) != 0:
        params = np.concatenate(params)   #making it  to a single array of individual numbers 
    
    # appending those values in fullmodel0 whose indices are not in paramix
    for i in range(len(fullmodel0)):
        if i in paramix:
            continue
        else:
            stable.append(fullmodel0[i])

    #making it  to a single array of individual numbers 
    stable = np.concatenate(stable)
    

    print("\n\n params =" , params)
    print("\n\n stable =" , stable)

    print("\n\n\n")


    #optimizing function
    if len(params) != 0:
        out = optimize.least_squares(ResidualUV, params, method=optmethod, 
                            verbose=2, max_nfev=10000, 
                            args=(stable, paramix, xyz,uv)) 

        
        if out.success is True:

            print('\noptimization is successfull')
            # print("\n\n out = ", out)

            #based on the conditions update the optimized values the fullmodel0 list

            if all(x in paramix for x in [0,2,3,4,5,6]):
                fullmodel0[0] = out.x[0:3]
                fullmodel0[2] = out.x[3:6]
                fullmodel0[3] = out.x[6:8]
                fullmodel0[4] = out.x[8:10]
                fullmodel0[5] = out.x[10:16]
                fullmodel0[6] = out.x[16:]
            

            elif all(x in paramix for x in [3,4,5,6]):
                fullmodel0[3] = out.x[0:2]
                fullmodel0[4] = out.x[2:4]
                fullmodel0[5] = out.x[4:10]
                fullmodel0[6] = out.x[10:]

            elif all(x in paramix for x in [0,2]):
                fullmodel0[0] = out.x[0:3]
                fullmodel0[2] = out.x[3:]
        
            else:
                fullmodel0[2] = out.x


            #defining a dictionary to store the new optimizd parametrs with their key
            cam_optimized = {'camera location': fullmodel0[0], 'image size': fullmodel0[1], 'view direction': fullmodel0[2], 'f':fullmodel0[3], 'c':fullmodel0[4], 'Rcoeff':fullmodel0[5], 'Tcoeff': fullmodel0[6], 'Full Model': fullmodel0}
        
            #Creating the optimized cam_object
            final_cam = const_cam_sec(cam_optimized['camera location'], cam_optimized['image size'], cam_optimized['view direction'], cam_optimized['f'],cam_optimized['c'],cam_optimized['Rcoeff'],cam_optimized['Tcoeff'])
            
            GCPxyz_proj1, depth, inframe = project(final_cam,xyz)

            print("\n\n GCPxyz_proj0 = ", GCPxyz_proj0)
            print("\n\n GCPxyz_proj1 = ", GCPxyz_proj1)
            print("\n\n uv = ", uv)

           
            if show == True:

                refimage = readImg(img_path)
                ims = refimage.shape

                #Plot GCPs using Utilities.plotResiduals function 
                plotResiduals(refimage, ims, uv, GCPxyz_proj0, GCPxyz_proj1)
        


            return final_cam
        
        elif out == 0:
            print("\n\nThe specified optimiation is not supported")
        
        else: 
            print('\n\nOptimization not done due to max iteration')
            return 
    else:
        print("\n\nNo optimization is specified")
        return 


    



    


    



def optimize_cam(cam,xyz,uv,freeparams,optmethod='trf'):
    import numpy as np
    from scipy import optimize
    from Camera import cam_constructor
    from residual_uv_upload import ResidualUV

    nanrows1 = np.array([np.any(np.isnan(row)) for row in xyz])
    nanrows2 = np.array([np.any(np.isnan(row)) for row in uv])


    #from xyz and uv delete those rows where nan values are there 
    xyz = np.delete(xyz, np.where(nanrows1), axis=0)
    uv = np.delete(uv, np.where(nanrows2), axis=0)

    #extract the full model data from the cam dictionary using the key
    fullmodel0 = cam['Full Model']

    #create a boolean array in which True is when the freeparams value is 1
   

    freeparams = [True if value == 1 else False for value in freeparams]
  
    paramix = np.where(freeparams)[0]  # Find indices of True elements
    

    params = []
    stable = []

    for i in paramix:
        params.append(fullmodel0[i])
    
    for i in range(len(fullmodel0)):
        if i in paramix:
            continue
        else:
            stable.append(fullmodel0[i])


    #optimizing function

    out = optimize.least_squares(ResidualUV, params, method=optmethod, 
                            verbose=2, max_nfev=5000, 
                            args=(stable, cam, xyz,uv)) 

    if out.success is True:

        print('\noptimization is successfull')
        for i in range(len(paramix)):
            fullmodel0[paramix[i]] = out.x[i]

        cam_optimized = {'camera location': fullmodel0[0:3], 'image size': fullmodel0[3:5], 'view direction': fullmodel0[5:8], 'f':fullmodel0[8:10], 'c':fullmodel0[10:12], 'Rcoeff':fullmodel0[12:18], 'Tcoeff': fullmodel0[18:], 'Full Model': fullmodel0}
        optimised_final_cam = {}
        optimised_final_cam = cam_constructor(cam_optimized['camera location'], cam_optimized['image size'], cam_optimized['view direction'], cam_optimized['f'])

        return optimised_final_cam
    else: 
        print('\nOptimization not done')
        return 

    


    





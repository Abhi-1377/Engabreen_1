def ResidualUV(params,stable,cam,xyz,uv):
    import numpy as np
    from Camera import project

    uv_projected,depth,inframe  = project(cam,xyz)
    
    residual = []

    for i in range(len(uv_projected)):
        residual.append(np.sqrt((uv_projected[i][0]-uv[i][0])**2 + 
                                (uv_projected[i][1]-uv[i][1])**2)) 

    residual = np.array(residual)#.reshape(-1,1) 

    return residual
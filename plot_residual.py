def plotResiduals(img, ims, gcp1, gcp2, gcp3):

    import matplotlib.pyplot as plt
    import numpy as np
    """Function to plot sets of points to show offsets. This is 
    commonly used for inspecting differences between image GCPs and projected 
    GCPs, e.g. within the optimiseCamera function
        
    Parameters
    ----------
    img : arr 
      Image array
    ims : list
      Image dimension (height, width)
    gcp1 : arr 
      Array with uv positions of image gcps
    gcp2 : arr 
      Array with initial uv positions of projected gcps
    gcp3 : arr 
      Array with optimised uv positions of projected gcps   
    """ 
    #Plot image                
    fig, (ax1) = plt.subplots(1)
    # fig.canvas.set_window_title('Average residual difference: ' + 
    #                             str(np.nanmean(gcp3-gcp2)) + ' px')
    ax1.axis([0,ims[1],ims[0],0])
    ax1.imshow(img, cmap='gray')
    
    #Plot UV GCPs
    ax1.scatter(gcp1[:,0], gcp1[:,1], color='red', marker='+', 
                label='UV')
    
    #Plot projected XYZ GCPs
    ax1.scatter(gcp2[:,0], gcp2[:,1], color='green', 
                marker='+', label='Projected XYZ (original)')

    #Plot optimised XYZ GCPs if given
    ax1.scatter(gcp3[:,0], gcp3[:,1], color='blue', 
                marker='*', label='Projected XYZ (optimised)')
    
    #Add legend and show plot
    ax1.legend()
    plt.show() 


 

# refimg=readImg(refimg)
# ims=refimg.shape  


def readImg(path, band='L', equal=True):
    from PIL import Image
    import numpy as np
    import operator
    from functools import reduce
    """Function to prepare an image by opening, equalising, converting to 
    either grayscale or a specified band, then returning a copy
    
    Parameters
    ----------
    path : str 
      Image file path directory
    band : str 
      Desired band output - 'R': red band; 'B': blue band; 'G': green band; 
      'L': grayscale (default='L')
    equal : bool 
      Flag to denote if histogram equalisation should be applied (default=True)

    Returns
    -------
    bw : arr
      Image array
    """  
    # Open image file
    band=band.upper()
    im=Image.open(path)
    
    #Equalise histogram
    if equal is True:
        h = im.convert("L").histogram()
        lut = []
    
        for b in range(0, len(h), 256):
    
            #Step size
            step = reduce(operator.add, h[b:b+256]) / 255
    
            #Create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i+b]
        
        #Convert to grayscale
        gray = im.point(lut*im.layers)    
    else:
        gray=im
    
    #Split bands if R, B or G is specified in inputs    
    if band=='R':
        gray,g,b=gray.split()
    elif band=='G':
        r,gray,b=gray.split() 
    elif band=='B':
        r,g,gray=gray.split() 
    else:
        gray = gray.convert('L')
    
    #Copy and return image    
    bw = np.array(gray).copy()
    
    return bw
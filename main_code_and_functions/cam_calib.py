def caliberate_cam():
  import numpy as np
  import cv2 as cv
  import glob
  import matplotlib.pyplot as plt

  #Termination criteria either when the accuracy reaches 0.001 or no.of iteration reaches 30
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  
  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  #Storing the image paths in a list 
  images = glob.glob('caliberation_images\*.JPG')

  #iterating through the images and finsing the chess board corners
  for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    #If corners found appending the object points to the list and finsing the corresponding imgg points
    if ret == True:
      objpoints.append(objp)
      corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners2)

  #Caliberating the camera using the img points, obj points and img
  ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  
  #Finding the error in caliberation, More accurate the error is close to 0
  mean_error = 0
  for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error


  #returning a dictionary containg all the calculated camera caliberation parameter
  return {'ret': ret, 'cam_mtx': mtx, 'distortion_coeff': dist, 'r_vectors': rvecs, 't_vectors': tvecs, 'mean_erroe': mean_error}

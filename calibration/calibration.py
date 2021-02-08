import numpy as np
import cv2
import os
#import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = []
#importing images from folder
#path # C:\aa\visual_odometry\calibration_nokia
path = 'C:\\aa\\visual_odometry'
#p = os.path.join(path, "image_0")
p = os.path.join(path, "calibration_nokia")
images = []
# taking images from folder #
for img in os.listdir(p):
    image = cv2.imread(os.path.join(p, img), cv2.IMREAD_GRAYSCALE)
    images.append(image)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
#print(objp*20)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i in range(0,9):
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(images[i], (9,7),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(images[i],corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(images[i], (9,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
cv2.destroyAllWindows()
ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, images[i].shape[::-1],None,None)
print(k)
print(dist)

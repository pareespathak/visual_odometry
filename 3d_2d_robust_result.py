import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

images = []
#importing images from folder

path = 'C:\\aa\\visual_odometry\\project\\KITTI_sample'
#p = os.path.join(path, "image_0")
p = os.path.join(path, "image_0")
images = []
# taking images from folder #
for img in os.listdir(p):
    image = cv.imread(os.path.join(p, img), cv.IMREAD_GRAYSCALE)
    images.append(image)

'''
cap = cv.VideoCapture('C:\\aa\\visual_odometry\\project\\L_shaped_path.avi')
ret, old_frame = cap.read()


i =0
while(i<=557):
    ret, old_frame = cap.read()
    image = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    cv.imshow('image',image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    #print(image.shape)
    images.append(image)
    i = i+1
cap.release()
cv.destroyAllWindows()

print(len(images))

'''
# calcOpticalFlowPyrLK function parameters
lk_params = dict( winSize  = (23,23),
              maxLevel = 2,
              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),minEigThreshold = 1e-4)

#Calibration Matrix

k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
'''
k = np.array([[518.56666108, 0., 329.45801792],
    [0., 518.80466479, 237.05589955],
    [  0., 0., 1.]])
'''

def track_features(img1, img2, corners):
    p1, st, err= cv.calcOpticalFlowPyrLK(img1, img2, corners,None,**lk_params)

    return p1,corners,st

def triangulaion(R,t,pt1,pt2,k):
    # projection matrix
    pr = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    pr_mat = np.dot(k,pr)
    P = np.hstack((R,t))
    P1 = np.dot(k,P)
    ch1 = np.array(pt1)
    ch2 = np.array(pt2)
    # making matrix 2xn :
    ch1 = pt1.transpose()
    ch2 = pt2.transpose()
    cloud = cv.triangulatePoints(pr_mat,P1,ch1,ch2)
    cloud = cloud[:4,:]
    div  = cloud[3,:] + 1e-8
    cloud = cloud / div
    #print(cloud)
    return cloud

def corners(img):
    #img1	=	cv.pyrDown(img,borderType = cv.BORDER_DEFAULT)
    corners = cv.goodFeaturesToTrack(img, mask = None, maxCorners = 400, qualityLevel = 0.01, minDistance = 8, blockSize = 21, useHarrisDetector = False, k = 0.05)
    return corners

pts1 = corners(images[0])
pts2,pts1,st = track_features(images[0],images[1],pts1)
dict1 = {}
dict2 = {}
#print(pts1.shape,pts2.shape)
for i in range(0,len(pts1)):
    dict1[tuple(pts1[i,0,:])] = tuple(pts2[i,0,:])

#print(dict1)
p1 = pts1[st == 1]
p2 = pts2[st == 1]
E, mask = cv.findEssentialMat(p2,p1,k,cv.RANSAC, prob=0.999,threshold = 0.4, mask=None)
# We select only inlier points
p1m = p1[mask.ravel()==1]
p2m = p2[mask.ravel()==1]
#Obtain rotation and translation for the essential matrix
# taking initial rotmat = I and translation mat = 000
r_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
t_init = np.array([[0],[0],[0]])
retval,R,t,mask=cv.recoverPose(E,p1m,p2m,k)
n_cloud = triangulaion(R,t,p1,p2,k)
points3d = n_cloud[:3,:]
points3d = points3d.transpose()
trans = t_init
rotation = np.dot(r_init,R)
#trans = np.dot(rotation,t)
#rotation = R
x1 = trans[0]    # taking x and z coordinates for graph
z1 = trans[2]
x = []
y = []
z = []
x.append(x1)
z.append(z1)
pts3,pts2,st1 = track_features(images[1],images[2],pts2)
'''
for i in range(0,len(pts2)):
    dict2[tuple(pts2[i,0,:])] = tuple(pts3[i,0,:])
print(dict2)
'''
rep = []
pic = []
# multiple images :
#print(points3d.shape)
flag = 0
count = 0
loop = 0
for i in range(1,150):
    #i = 5*j
    if flag == 1:
        pts1 = corners(images[i-1])
        # tracking them in frame 2:
        pts2,pts1,st = track_features(images[i-1],images[i],pts1)
        p1 = pts1[st == 1]
        p2 = pts2[st == 1]
        n_cloud = triangulaion(rmat,tvec,p1,p2,k)
        points3d = n_cloud[:3,:]
        points3d = points3d.transpose()
        pts3,pts2,st1 = track_features(images[i],images[i+1],pts2)
        flag = 0
    else:
        pts3,pts2,st1 = track_features(images[i],images[i+1],pts2)
    p3 = pts3[st == 1]
    #print(pts3.shape,pts2.shape,p3.shape)
    retval, rvec, tvec, inliers	= cv.solvePnPRansac(points3d,p3,k, distCoeffs = None, useExtrinsicGuess = True,iterationsCount = 100,
    reprojectionError = 5.0, confidence = 0.99, flags = cv.SOLVEPNP_EPNP)
    #print(inliers.shape)
    kpre ,_= cv.projectPoints(points3d,rvec,tvec.T,k,distCoeffs = np.zeros((5,1)))
    kpre = kpre.reshape(kpre.shape[0],2)
    #print(kpre.shape, p3.shape)
    rep_error = np.linalg.norm((kpre-p3),axis = 1)
    rep_error = np.linalg.norm(rep_error)/len(rep_error)
    rep.append(rep_error)
    pic.append(i)
    print(rep_error)
    rmat, jacobian = cv.Rodrigues(rvec)
    p1T = pts2[st1 == 1]
    p2T = pts3[st1 == 1]
    n_cloud = triangulaion(rmat,tvec,p1T,p2T,k)
    points3d = n_cloud[:3,:]
    points3d = points3d.transpose()
    thresh = 5
    if rep_error <= 8:
        rotation = np.dot(rotation,rmat)
        t1 = np.linalg.norm(tvec, axis = 0)
        tvec1 = tvec/t1
        trans = trans - np.dot(rotation,tvec1)
        x1 = trans[0]
        z1 = trans[2]
        x.append(x1)
        z.append(-1*z1)
        loop = loop +1

    st = st1
    pts2 = pts3
    if rep_error >= thresh :
        #print(i)
        flag = 1
        count = count + 1

x_truth = []
z_truth = []
print("count", count)
print("loop", loop, "i",i)
print("sum", loop+i)
# ground truth using pose doc
ground_truth = np.loadtxt('C:\\aa\\visual_odometry\\project\\KITTI_sample\\poses.txt')
x_truth=[]
z_truth=[]
for i in range(150):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11])
#plt.plot(x_truth,z_truth, label = "ground_truth")
plt.plot(x,z,color='green',label = "plotted trajectory")
#plt.scatter(x,z,c = 'green', s = 3)
#print(len(pic))
refline = np.zeros(len(pic),dtype = int)
refline = refline + 5

#plt.plot(pic,refline,color = 'blue')
#plt.plot(pic,rep,color = 'green',label = "reprojection ")
#plt.scatter(pic,rep,c = "green", s=3)
plt.title("monocular camera based plot")

#print(z_truth)
plt.show()

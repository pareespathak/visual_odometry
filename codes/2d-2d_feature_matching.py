#importing packages 
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

'''
importing images from folder
'''

#path = 'C:\\aa\\visual_odometry\\project\\KITTI_sample'      # path of the image folder
path = 'visual_odometry'
#p = os.path.join(path, "images")                             # image folder name
p = os.path.join(path, "dataset")                             # image folder name
images = []
for img in os.listdir(p):
    image = cv.imread(os.path.join(p, img), cv.IMREAD_GRAYSCALE)
    images.append(image)

#Calibration Matrix for KITTI sample
k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


'''
#importing video as image frames

#On board dataset

cap = cv.VideoCapture('C:\\aa\\visual_odometry\\project\\L_shaped_path.avi')   #folder location and video name.format
ret, old_frame = cap.read()
images = []
i =0
while(i<=550):
    ret, old_frame = cap.read()
    image = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    #cv.imshow('image',image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    images.append(image)
    i = i+1
#cap.release()
#cv.destroyAllWindows()
#print(len(images))

#calibration matrix for video camera (from Zhang's method)

k = np.array([[518.56666108, 0., 329.45801792],
    [0., 518.80466479, 237.05589955],
    [  0., 0., 1.]])
'''

# tracking features using SIFT 
def track_features(img1,img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT 
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    # FINDING INDEX PARAMETERS FOR FLANN OPERATORS
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    match = flann.knnMatch(des1,des2,k=2)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper to estimate correspondence 
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    #print(pts1.shape)
    return pts1,pts2

# triangulation between image frames to estimate 3d point cloud 
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
    #print(cloud.shape)
    cloud = cloud[:4,:]
    return cloud

# relative scale calculation 
def scale(O_cloud,N_cloud):
    siz = min(O_cloud.shape[1],N_cloud.shape[1]) 
    o_c = np.zeros((3,siz))
    n_c = np.zeros((3,siz))
    o_c = O_cloud[:,:siz]
    n_c = N_cloud[:,:siz]
    # axis one and shift = 1 == rolling column by one
    o_c1 = np.roll(o_c, axis=1,shift = 1)
    n_c1 = np.roll(n_c, axis=1,shift = 1)
    # axis = 0 == taking norm along column
    scale = np.linalg.norm((o_c - o_c1), axis = 0)/(np.linalg.norm(n_c - n_c1, axis = 0) + 1e-8 )
    # taking median along the row (norm)
    scale = np.median(scale)
    return scale

###### for first two images
pts1,pts2 = track_features(images[0],images[1])
# Essential matrix with RANSAC outlier rejection
E, mask = cv.findEssentialMat(pts2,pts1,k,cv.RANSAC, prob=0.999,threshold = 0.4, mask=None)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
#Obtain rotation and translation for the essential matrix
# taking initial rotation matrix as identity matrix = I and translation mat = [0,0,0]
r_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
t_init = np.array([[0],[0],[0]])
retval,R,t,mask=cv.recoverPose(E,pts1,pts2,k)
n_cloud = triangulaion(r_init,t_init,pts1,pts2,k)
trans = np.dot(R,t)
rotation = R
x1 = trans[0]    # x and z coordinates for graph
z1 = trans[2]
x = []
y = []
z = []
x.append(x1)
z.append(z1)
# multiple frames
for i in range(1,225):
    #j = 2*i             #for skipping some frames in video datasets            
    o_cloud = n_cloud
    pts1,pts2 = track_features(images[i],images[i+1])
    #Essential matrix estimation 
    E, mask = cv.findEssentialMat(pts2,pts1,k,cv.RANSAC, prob=0.999,threshold = 0.4 ,mask=None)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    #Obtain rotation and translation for the essential matrix
    retval,R,t,mask=cv.recoverPose(E,pts1,pts2,k)
    n_cloud = triangulaion(R,t,pts1,pts2,k)
    #print(o_cloud,n_cloud)
    sc = scale(o_cloud, n_cloud)
    #print(sc)
    trans = trans - sc*np.dot(rotation,t)  #scale fac
    rotation = np.dot(rotation,R)
    x1 = trans[0]
    z1 = trans[2]
    x.append(x1)
    z.append(z1)

x_truth = []
z_truth = []

# ground truth using pose doc
ground_truth = np.loadtxt('C:\\aa\\visual_odometry\\project\\KITTI_sample\\poses.txt')  #path of ground truth 
x_truth=[]
z_truth=[]
for i in range(150):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11])

# Results and trajectory 
plt.plot(x_truth,z_truth, label = "ground_truth")
plt.plot(x,z,color='green',label = "plotted trajectory")
#plt.scatter(x,z,c='green',s=4)
plt.legend()
plt.xlabel("y")
plt.ylabel("z")
plt.title("Results(2d-2d_video dataset_feature_matching)")
plt.show()

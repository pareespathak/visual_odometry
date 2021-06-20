import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


#importing images from folder

#path = 'C:\\aa\\visual_odometry\\project\\KITTI_sample'
#p = os.path.join(path, "image_0")
path = 'visual_odometry'
p = os.path.join(path, "dataset")
images = []
for img in os.listdir(p):
    image = cv.imread(os.path.join(p, img), cv.IMREAD_GRAYSCALE)
    images.append(image)

#Calibration Matrix
k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
'''
images = []
cap = cv.VideoCapture('C:\\aa\\visual_odometry\\project\\L_shaped_path.avi')
ret, old_frame = cap.read()
images = []
i =0
while(i<=550):
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

k = np.array([[518.56666108, 0., 329.45801792],
    [0., 518.80466479, 237.05589955],
    [  0., 0., 1.]])
'''
# calcOpticalFlowPyrLK function parameters
lk_params = dict( winSize  = (15,15),
              maxLevel = 5,
              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03), minEigThreshold = 1e-4)

def corners(img):
    #img1	=	cv.pyrDown(img,borderType = cv.BORDER_DEFAULT)
    corners = cv.goodFeaturesToTrack(img, mask = None, maxCorners = 400, qualityLevel = 0.01, minDistance = 7, blockSize = 3, useHarrisDetector = False, k = 0.04  )
    #print(corners.shape)
    #print(corners.shape,corners)
    return corners


def track_features(img1, img2, corners):
    p1, st, err= cv.calcOpticalFlowPyrLK(img1, img2, corners,None,**lk_params)
    p1 = p1[st == 1]
    corners = corners[st == 1]
    return p1,corners

# triangulation
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
    div  = cloud[3,:] + 1e-8
    cloud = cloud / div
    return cloud

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

# image 1 and 2
pts1 = corners(images[0])
c1 = pts1
pts2,pts1 = track_features(images[0],images[1],pts1)
E, mask = cv.findEssentialMat(pts2,pts1,k,cv.RANSAC, prob=0.999,threshold = 0.4, mask=None)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
#Obtain rotation and translation for the essential matrix
# taking initial rotmat = I and translation mat = 000
r_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
t_init = np.array([[0],[0],[0]])
retval,R,t,mask=cv.recoverPose(E,pts1,pts2,k)
n_cloud = triangulaion(r_init,t_init,pts1,pts2,k)
trans = np.dot(R,t)
rotation = R
x1 = trans[0]    # taking x and z coordinates for graph
z1 = trans[2]
x = []
y = []
z = []
x.append(x1)
z.append(z1)

# multiple frames
for j in range(1,250):
    o_cloud = n_cloud
    pts1 = c1
    pts2,pts1 = track_features(images[j],images[j+1],pts1)
    #print(pts2.shape)
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
    if pts2.shape[0] <= 395:
        c1 = corners(images[j])

# ground truth using pose doc
ground_truth = np.loadtxt('C:\\aa\\visual_odometry\\project\\KITTI_sample\\poses05.txt')
x_truth=[]
z_truth=[]
for i in range(250):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11])
plt.plot(x_truth,z_truth, label = "ground_truth")
plt.xlabel("x")
plt.ylabel("z")
plt.plot(x,z,color='green',label = "plotted trajectory")
#plt.scatter(x,z,c='green', s=4)
plt.legend()
plt.title("Results: 2d-2d_tracking_KITTI_05_Dataset")
plt.show()
plt.savefig('2d-2d-euc.png')
cv.destroyAllWindows()

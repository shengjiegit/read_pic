import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from skimage import data,color,morphology,feature
import cv2

img=np.array(Image.open('E:\RadMosaicNCN(2010-)\Rad_ANCN_201707\Rad_ANCN_20170708\CREF\ANCN.CREF000.20170708.121800.PNG').convert('RGB'))
img = img[::-1,0:700,:]
thresholdList = [(173,144,240),(150,0,180),(255,0,240),(192,0,0),(214,0,0),(255,0,0)]#(255,144,0),(231,192,0)
for i in thresholdList:
   indexx=(img[:, :, 0] == i[0]) & (img[:, :, 1] == i[1]) & (img[:, :, 2] == i[2])
 # img[(img[:,:,0]==x[0])&(img[:,:,1]==x[1])&(img[:,:,2]==x[2])]=[0, 0, 0]
   Y,X=np.where(indexx==True)
   img[Y,X]=[0,0,0]
#vv = np.concatenate((np.array(X), np.array(Y)))
vv=list(zip(Y.tolist(),X.tolist()))
print(img.shape)
plt.imshow(img) # 显示图片
plt.xlim((0, 800))
plt.ylim((0, 600))
plt.show()
#x=np.arange.img.shape[0]
#y=np.arange.img.shape[1]
db=DBSCAN(eps =8, min_samples =4).fit_predict(vv)
print(db)
maxdbz=max(set(db.tolist()), key=db.tolist().count)
dbz=db[db==maxdbz]
Xdbz=X[db==maxdbz]
Ydbz=Y[db==maxdbz]
#db=DBSCAN(eps=1,min_samples=20, algorithm='ball_tree',metric='euclidean').fit_predict(vv)
#print(X.shape)
#print(Xdbz.shape)
plt.xlim((0, 800))
plt.ylim((0, 600))
plt.axis('off') # 不显示坐标轴
plt.scatter(Xdbz,Ydbz,c=dbz,marker='o',edgecolors = 'none')
#plt.axis('off')
plt.savefig('d:/1.jpg')
#plt.legend(db)
plt.show()
#imgg=np.array(Image.open('D:/1.jpg').convert('L'))
#edgs=feature.canny(imgg, sigma=3, low_threshold=10, high_threshold=50)
#chull = morphology.convex_hull_object(edgs)
imgg=cv2.imread('D:/1.jpg')
thresh = cv2.Canny(imgg,500,800)
#gray=cv2.cvtColor(mapimg,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(gray,127,255,0)   #进行二值化
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

ellipse = cv2.fitEllipse(contours[1])
imgg = cv2.ellipse(imgg,ellipse,(0,255,0),2)
#imgg = cv2.drawContours(imgg, contours, -1, (0,255,0), 3)
cv2.imshow('img',imgg)
cv2.waitKey()
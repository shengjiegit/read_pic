from sklearn.cluster import DBSCAN
from skimage import data,color,morphology,feature
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
img=np.array(Image.open('D:\wujia\华北雷达平图\Rad_ANCN_201706\Rad_ANCN_20170602\CREF\ANCN.CREF000.20170602.114800.PNG').convert('L'))
edgs=feature.canny(img, sigma=3, low_threshold=10, high_threshold=50)
chull = morphology.convex_hull_object(edgs)
plt.imshow(chull)
plt.show()

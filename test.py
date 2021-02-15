import json
import ast
import cv2
import numpy as np
import micasense.imageutils as imageutils
B = 0
G = 1
R = 2
a1 = cv2.imread('./tmp/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_1.tif',cv2.IMREAD_LOAD_GDAL)
a2 = cv2.imread('./tmp/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_2.tif',cv2.IMREAD_LOAD_GDAL)
a3 = cv2.imread('./tmp/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_3.tif',cv2.IMREAD_LOAD_GDAL)
im_display = np.zeros((a1.shape[0],a1.shape[1],3), dtype=np.float32 )
im_display[:,:,B] = a1
im_display[:,:,G] = a2
im_display[:,:,R] = a3
min = np.min(im_display)
max = np.max(im_display)
print(min, max)
cv2.imshow("before",im_display)

im_min = np.percentile(im_display[:,:,:].flatten(), 0.5)  # modify these percentiles to adjust contrast
im_max = np.percentile(im_display[:,:,:].flatten(), 99.5)
print(im_min , im_max)
im_display[:,:,B] = imageutils.normalize(a1, im_min, im_max)
im_display[:,:,G] = imageutils.normalize(a2, im_min, im_max)
im_display[:,:,R] = imageutils.normalize(a3, im_min, im_max)
cv2.imshow("a1",im_display)


im_display = im_display *255
TGI=0.5*((150*(R-G))-(100*(R-B)))
MIN=0.5*((150*(-255))-(25500))
MAX=0.5*((150*(255))-(-25500))
print(MIN)
print(MAX)

TGI=0.5*((150*(im_display[:,:,R]-im_display[:,:,G]))-(100*(im_display[:,:,R]-im_display[:,:,B])))
TGI2 = imageutils.normalize(TGI, MIN, MAX)
cv2.imshow("tgi2",TGI2)

min = np.min(TGI)
max = np.max(TGI)
print(min , max)
TGI = TGI - min
max = np.max(TGI)
TGI =(TGI /max)

cv2.imshow("tgi",TGI)


cv2.waitKey(0)
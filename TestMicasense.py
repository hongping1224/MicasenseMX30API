import cv2
import numpy as np
import micasense.capture as cap
import micasense.imageutils as imageutils
import os, glob
from Allignment import AllignImage, GetAllignmentMatrix, SaveAllignmentMatrix, ReadAllignmentMatrix
from ops import NDVI
import time
from main import NormalizeAndDrawLegend
def test_Allignment(i , o):
    imagePath = i.replace("_1.tif","_*.tif")
    imageNames = glob.glob(imagePath)
    capture = cap.Capture.from_filelist(imageNames)

    allignmat, havePrev = ReadAllignmentMatrix(".")
    if havePrev == False:
        allignmat=GetAllignmentMatrix(capture)
        SaveAllignmentMatrix("a_mat_{}.txt",allignmat)
    start = time.time()
    im_aligned = AllignImage(allignmat,capture)
    print(f"Allignment time {time.time()-start}")
    #rgb_band_indices = [capture.band_names().index('Red'),capture.band_names().index('Green'),capture.band_names().index('Blue')]
    # rgb_band_indices = [capture.band_names().index('Blue'),capture.band_names().index('Green'),capture.band_names().index('Red')]
    # cir_band_indices = [capture.band_names().index('NIR'),capture.band_names().index('Red'),capture.band_names().index('Green')]
    # im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],im_aligned.shape[2]), dtype=np.float32 )
    ndvi = NDVI(im_aligned)
    ndvi = NormalizeAndDrawLegend(ndvi,0,1)
    cv2.imshow("NDVI",cv2.resize(ndvi,(int(ndvi.shape[1]/2),int(ndvi.shape[0]/2))))
    cv2.waitKey(1)
    # for i in rgb_band_indices:
    #     im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i])#, im_min, im_max)
    # rgb = im_display[:,:,rgb_band_indices]
    # for i in cir_band_indices:
    #     im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i])
    # cir = im_display[:,:,cir_band_indices]
    # rgb= cv2.normalize(rgb,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)
    # #cv2.imshow("t",rgb)
    # #cv2.waitKey(0)
    # cir= cv2.normalize(cir,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)
    # # cv2.imwrite(outrgbPath,rgb)
    # cv2.imwrite(outcirPath,cir)
    # for i in range(im_aligned.shape[2]):
    #    cv2.imwrite(outPath.format(i+1),cv2.normalize(im_aligned[:,:,i],None,0.,1.,cv2.NORM_MINMAX,cv2.CV_32F))
    # return


def main():
    if os.path.exists("./input") == False:
        os.mkdir("./input")
        print("input folder doesn't exist, create a new input folder")

    allpath = glob.glob('input/*_1.tif')
    outpath = "./output/"
    if os.path.exists(outpath) == False:
        os.mkdir(outpath)
    if os.path.exists(os.path.join(outpath,"rgb")) == False:
        os.mkdir(os.path.join(outpath,"rgb"))
    if os.path.exists(os.path.join(outpath,"cir")) == False:
        os.mkdir(os.path.join(outpath,"cir"))
    total = len(allpath)
    if total ==0:
         print("There is no imageset in input folder")
         print("Make sure Imageset naming is in xxxx_1.tif , xxxx_2.tif, xxxx_3.tif ,xxxx_4.tif ,xxxx_5.tif")
         return

    i = 1
    for p in allpath:
        print(p , "(",i,"/",total,")")
        i= i+1
        test_Allignment(p,outpath)
        '''
        try:
            test_Allignment(p,outpath)
        except:
            pass
        '''
    allignmentfile = glob.glob('./a_mat_*')
    for al in allignmentfile :
        break
        os.remove(al)

if __name__ == "__main__":
    main()
   
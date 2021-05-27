import cv2
import numpy as np
import micasense.imageutils as imageutils

B = 0
G = 1
R = 2
NIR =3
REDEDGE=4

def NDVI(im):
    NDVI = (im[:,:,NIR]-im[:,:,R])/(im[:,:,NIR]+im[:,:,R])
    return NDVI

def NBI(im):
    norm = np.zeros(im.shape)
    im_sum = (im[:,:,0]+im[:,:,1]+im[:,:,2]+im[:,:,3]+im[:,:,4])
    for i in range(im.shape[2]):
        norm[:,:,i] = im[:,:,i]/im_sum
    OSAVI =  (1+0.16) * (norm[:,:,3]-norm[:,:,2])/(norm[:,:,3]+norm[:,:,2]+0.16)
    TCARI = 3* ((norm[:,:,4]-norm[:,:,2])-
            (0.2*(norm[:,:,4]-norm[:,:,1])*
                (norm[:,:,4]/norm[:,:,2])))
    NBI = TCARI/OSAVI
    return NBI

def CIR(im):
    cir = im[:,:,[NIR,R,G]]
    return cir

def RGB(im):
    rgb = im[:,:,[B,G,R]]
    
    rgb[:,:,B] = imageutils.normalize(rgb[:,:,B])#, im_min, im_max)
    rgb[:,:,G] = imageutils.normalize(rgb[:,:,G])#, im_min, im_max)
    rgb[:,:,R] = imageutils.normalize(rgb[:,:,R])#, im_min, im_max)
    return rgb

def rawRGB(im):
    rgb = im[:,:,[B,G,R]]
    return rgb

def TGI(im):
    TGI=0.5*((150*(im[:,:,R]-im[:,:,G]))-(100*(im[:,:,R]-im[:,:,B])))
    TGI = (TGI+5000)/10000
    return TGI
import numpy as np
import os, glob
import micasense.capture as cap
import micasense.imageutils as imageutils
import cv2
from ops import RGB,CIR,NDVI,NBI,TGI,NDRE


def ReadAllignmentMatrix(path):
    matrix = []
    matrix_path = glob.glob(os.path.join(path,'a_mat_*.txt'))
    matrix_path.sort()
    if len(matrix_path) == 0:
        return [] , False
    for path in matrix_path:
        try:
            matrix.append(np.loadtxt(path))
        except:
            print("notfound")
            return [] ,False
    return matrix , True

def SaveAllignmentMatrix(path, matrix):
    for i in range(len(matrix)):
       np.savetxt(path.format(i),matrix[i])
    return

def AllignImage(mat, images):
    if images.dls_present():
        img_type='reflectance'
    else:
        img_type = "radiance"
    warp_mode = cv2.MOTION_HOMOGRAPHY
    match_index = 0
    cropped_dimensions, _ = imageutils.find_crop_bounds(images, mat, warp_mode=warp_mode)
    np.array([0],dtype=np.float32)
    im_aligned = imageutils.aligned_capture(images, mat, warp_mode, cropped_dimensions, match_index, img_type=img_type)
    return im_aligned

def GetAllignmentMatrix(images,iteration = 20):
    ## Alignment settings
    match_index = 4 # Index of the band, here we use green
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = 3 # for 10-band imagery we use a 3-level pyramid. In some cases
    print("Calculating")
    warp_matrices, alignment_pairs = imageutils.align_capture(images,
                                                          ref_index = match_index,
                                                          max_iterations = iteration,
                                                          warp_mode = warp_mode,
                                                          pyramid_levels = pyramid_levels)
    print("Done")
    return warp_matrices

def GetAllignmentMatrix2(images):
    ## Alignment settings
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    print("Calculating")
    im = [] 
    for img in images.images:
        im.append(img.undistorted_radiance())
    warp_matrices = OrbAllignAll(im)
    print("Done")
    return warp_matrices


def allignmentMatrixTostring(mat):
    s = []
    for m in mat:
        s.append(m.tolist())
    return s

def loadfromstring(s):
    rmat = []
    for ss in s:
        rmat.append(np.array(ss,dtype=float))
    return rmat



def OrbAllign(im1,im2):
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    im1Gray =im1.astype(np.uint8)# cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray =im2.astype(np.uint8)# cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches = sorted(matches,key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return h

def  OrbAllignAll(images):
    imReference = images[0]
    # Read reference image
    imReference = np.uint8(imageutils.normalize(imReference, np.min(imReference), np.max(imReference))*255)
    mat = [] 
    mat.append(np.eye(3))
    for i in range(1,len(images)):
        print(i)
        im = images[i]
        im = np.uint8(imageutils.normalize(im, np.min(im), np.max(im))*255)
        h = OrbAllign( imReference,im)
        mat.append(h)
    return mat



def AutoAllign():
    result = glob.glob("./input/*_5.tif")
    suffix = "_5.tif"
    files = []
    for f in result:
        filename = f.replace(suffix,"_*.tif")
        files.append(filename)
    files.sort()  
    print("Allign")
    for i in range(len(files)):
        # try:
            print(i)
            result = glob.glob(files[len(files)-(i+1)])     
            print(result)
            capture = cap.Capture.from_filelist(result)
            mat = GetAllignmentMatrix(capture)
            SaveAllignmentMatrix("a_mat_{}.txt",mat)
            im = AllignImage(mat,capture)
            rgb_band_indices = [capture.band_names().index('Red'),capture.band_names().index('Green'),capture.band_names().index('Blue')]
            cv2.imshow("result",im[:,:,rgb_band_indices])
            cv2.waitKey(0)
            print("success")
            break   
        # except :
        #     continue
    return mat


def main():
    import tifffile
    path = "./"
    confPath = ""
    # mat = AutoAllign()
    mat,success= ReadAllignmentMatrix(path)
    if success == False:
        print("read mat failed")
        return
    for i in glob.glob("./input/*_5.tif"):
        imgs =  glob.glob(i.replace("_5.tif","_*.tif"))
        capture = cap.Capture.from_filelist(imgs)
        im = AllignImage(mat,capture)
        cv2.imwrite(os.path.join("output/rgb",os.path.basename(i).replace("_5.tif",".png")),(RGB(im)*255).astype(np.uint8))
        cv2.imwrite(os.path.join("output/ndvi",os.path.basename(i).replace("_5.tif",".png")),(NDVI(im)*255).astype(np.uint8))
        tifffile.imwrite(os.path.join("output/",os.path.basename(i).replace(".tif",".tif")),im)
    #SaveAllignmentMatrix(confPath, mat)
    return

if __name__== "__main__":
    main()

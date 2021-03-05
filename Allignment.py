import numpy as np
import os, glob
import micasense.capture as cap
import micasense.imageutils as imageutils
import cv2

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


def AutoAllign():
    result = glob.glob("./img/*_5.tif")
    suffix = "_5.tif"
    files = []
    for f in result:
        filename = f.replace(suffix,"_*.tif")
        files.append(filename)
    files.sort()  
    print("Allign")
    for i in range(len(files)):
        try:
            print(i)
            result = glob.glob(files[len(files)-(i+1)])     
            capture = cap.Capture.from_filelist(result)
            mat = GetAllignmentMatrix(capture)
            SaveAllignmentMatrix("a_mat_{}.txt",mat)
            break   
        except :
            continue
    return

def AllignImage(mat, images):
    if images.dls_present():
        img_type='reflectance'
    else:
        img_type = "radiance"
    warp_mode = cv2.MOTION_HOMOGRAPHY
    match_index = 4
    cropped_dimensions, _ = imageutils.find_crop_bounds(images, mat, warp_mode=warp_mode)
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
                                                          max_iterations = maxiteration,
                                                          warp_mode = warp_mode,
                                                          pyramid_levels = pyramid_levels)
    print("Done")
    return warp_matrices

def GetAllignmentMatrix2(images):
    ## Alignment settings
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    print("Calculating")
    warp_matrices = OrbAllignAll(images)
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
        rmat.append(np.array(ss))
    return rmat

def main():
    path = "./"
    confPath = ""
    mat,a = ReadAllignmentMatrix(path)
    s = allignmentMatrixTostring(mat)
    rmat = loadfromstring(s)
    for i in range(len(mat)):
        print(np.sum(rmat[i]-mat[i]))
    #SaveAllignmentMatrix(confPath, mat)
    return


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
    matches.sort(key=lambda x: x.distance, reverse=False)

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

def OrbAllignAll(capture):
    imReference = capture.images[0].undistorted_radiance()
    # Read reference image
    imReference = imageutils.normalize(imReference, 0, np.max(imReference))*255
    mat = [] 
    mat.append(np.eye(3))
    for i in range(1,5):
        im = capture.images[i].undistorted_radiance()
        im = imageutils.normalize(im, 0, np.max(im))*255
        h = OrbAllign( imReference,im)
        mat.append(h)
    return mat


if __name__== "__main__":
    main()

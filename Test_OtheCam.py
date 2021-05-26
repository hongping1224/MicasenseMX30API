import os
import cv2
import tifffile as tiff
from Allignment import AllignImage, OrbAllignAll, SaveAllignmentMatrix, ReadAllignmentMatrix
from micasense import imageutils
from micasense.imageutils import Bounds
import numpy as np

def main():
    rootpath = "C:\\Users\\HongPing\\Desktop\\download_2021-05-25_20-50-34\\農試所多光譜相機"
    imgs = [os.path.join(rootpath,f) for f in os.listdir(rootpath) if os.path.isfile(os.path.join(rootpath, f))]
    print(imgs)
    im = []
    maxx ,maxy = 0,0
    for img in imgs:
        tmpim = tiff.imread(img)
        maxx = max(tmpim.shape[1],maxx)
        maxy = max(tmpim.shape[0],maxy)
        if len(tmpim.shape) > 2:
            for i in range (tmpim.shape[2]):
                im.append(tmpim[:,:,i].copy())
        else:
            im.append(tmpim.copy())
        del tmpim
    print(maxx , maxy)

    for i in range(len(im)):
        im[i] = cv2.resize(im[i],(maxx,maxy),interpolation= cv2.INTER_NEAREST)

    allignmat, havePrev = ReadAllignmentMatrix(".")
    if havePrev == False:
        allignmat=OrbAllignAll(im)
        SaveAllignmentMatrix("a_mat_{}.txt",allignmat)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    match_index = 0
    print(f"band count : {len(im)}")
    distortion_coof = [np.array([0,0,0,0,0]) for f in range(len(im))]
    #RGBfocallength = 7.2mm
    #monofocallength = 6mm
    #cx = 2592/2 #cy = 1944/2
    #fx = 
    #sensor size =4.8×3.6mm(黑白相機)5.7×4.3mm(RGB相機)
    #[[fx,0,cx],[0,fy,cy],[0,0,1]]
    camera_matrix = [np.array([[7.2*(2592/5.7),0,2592/2],[0,7.2*(1944/4.3),1944/2],[0,0,1]]) for f in range(3)]
    camera_matrix_mono = [np.array([[6.0*(2592/4.8),0,2592/2],[0,6.0*(1944/3.6),1944/2],[0,0,1]]) for f in range(4)]
    for m in camera_matrix_mono:
        camera_matrix.append(m)
    cropped_dimensions, _ = find_crop_bounds(im, allignmat,camera_matrix,distortion_coof, warp_mode=warp_mode)
    print(cropped_dimensions)
    im_aligned = aligned_capture(im, allignmat, cropped_dimensions)
    shape = im_aligned.shape
    print(shape)
    outPath = "output_{}.tiff"
    for i in range(im_aligned.shape[2]):
        cv2.imwrite(outPath.format(i+1),cv2.normalize(im_aligned[:,:,i],None,0.,1.,cv2.NORM_MINMAX,cv2.CV_32F))
        cv2.imshow(str(i),cv2.resize(imageutils.normalize(im_aligned[:,:,i]),(int(shape[1]/2),int(shape[0]/2))))
    cv2.waitKey(0)


#apply homography to create an aligned stack
def aligned_capture(capture, warp_matrices, cropped_dimensions,interpolation_mode=cv2.INTER_LANCZOS4):
    width, height = (capture[0].shape[1],capture[0].shape[0])

    im_aligned = np.zeros((height,width,len(warp_matrices)), dtype=np.float32 )

    for i in range(0,len(warp_matrices)):
        img = capture[i]
        im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                warp_matrices[i],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
    (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
    im_cropped = im_aligned[top:top+h, left:left+w][:]
    return im_cropped

def find_crop_bounds(capture,registration_transforms,camera_matrix,distortion_coof,warp_mode=cv2.MOTION_HOMOGRAPHY):
    """Compute the crop rectangle to be applied to a set of images after
    registration such that no pixel in the resulting stack of images will
    include a blank value for any of the bands

    Args:

    capture- an image capture

    registration_transforms - a list of affine transforms applied to
    register the image. It is required.

    returns the left,top,w,h coordinates  of the smallest overlapping rectangle
    and the mapped edges of the images
    """
    image_sizes = [(image.shape[1],image.shape[0]) for image in capture]
    lens_distortions =distortion_coof
    camera_matrices =  camera_matrix

    bounds = [get_inner_rect(s, a, d, c,warp_mode=warp_mode)[0] for s, a, d, c in zip(image_sizes,registration_transforms, lens_distortions, camera_matrices)]
    edges = [get_inner_rect(s, a, d, c,warp_mode=warp_mode)[1] for s, a, d, c in zip(image_sizes,registration_transforms, lens_distortions, camera_matrices)]
    combined_bounds = get_combined_bounds(bounds, image_sizes[0])

    left = np.ceil(combined_bounds.min.x)
    top = np.ceil(combined_bounds.min.y)
    width = np.floor(combined_bounds.max.x - combined_bounds.min.x)
    height = np.floor(combined_bounds.max.y - combined_bounds.min.y)
    return (left, top, width, height),edges

def get_inner_rect(image_size, affine, distortion_coeffs, camera_matrix,warp_mode=cv2.MOTION_HOMOGRAPHY):
    w = image_size[0]
    h = image_size[1]

    left_edge = np.array([np.ones(h)*0, np.arange(0, h)]).T
    right_edge = np.array([np.ones(h)*(w-1), np.arange(0, h)]).T
    top_edge = np.array([np.arange(0, w), np.ones(w)*0]).T
    bottom_edge = np.array([np.arange(0, w), np.ones(w)*(h-1)]).T

    left_map = map_points(left_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    left_bounds = min_max(left_map)
    right_map = map_points(right_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    right_bounds = min_max(right_map)
    top_map = map_points(top_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    top_bounds = min_max(top_map)
    bottom_map = map_points(bottom_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    bottom_bounds = min_max(bottom_map)

    bounds = Bounds()
    bounds.max.x = right_bounds.min.x
    bounds.max.y = bottom_bounds.min.y
    bounds.min.x = left_bounds.max.x
    bounds.min.y = top_bounds.max.y
    edges = (left_map,right_map,top_map,bottom_map)
    return bounds,edges

def get_combined_bounds(bounds, image_size):
    w = image_size[0]
    h = image_size[1]

    final = Bounds()

    final.min.x = final.min.y = 0
    final.max.x = w
    final.max.y = h

    for b in bounds:
        final.min.x = max(final.min.x, b.min.x)
        final.min.y = max(final.min.y, b.min.y)
        final.max.x = min(final.max.x, b.max.x)
        final.max.y = min(final.max.y, b.max.y)

    # limit to image size
    final.min.x = max(final.min.x, 0)
    final.min.y = max(final.min.y, 0)
    final.max.x = min(final.max.x, w-1)
    final.max.y = min(final.max.y, h-1)
    # Add 1 px of margin (remove one pixel on all sides)
    final.min.x += 1
    final.min.y += 1
    final.max.x -= 1
    final.max.y -= 1

    return final

def min_max(pts):
    bounds = Bounds()
    for p in pts:
        if p[0] > bounds.max.x:
            bounds.max.x = p[0]
        if p[1] > bounds.max.y:
            bounds.max.y = p[1]
        if p[0] < bounds.min.x:
            bounds.min.x = p[0]
        if p[1] < bounds.min.y:
            bounds.min.y = p[1]
    return bounds

def map_points(pts, image_size, warpMatrix, distortion_coeffs, camera_matrix,warp_mode=cv2.MOTION_HOMOGRAPHY):
    # extra dimension makes opencv happy
    pts = np.array([pts], dtype=np.float)
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image_size, 1)
    new_pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs, P=new_cam_mat)
    if warp_mode == cv2.MOTION_AFFINE:
        new_pts = cv2.transform(new_pts, cv2.invertAffineTransform(warpMatrix))
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        new_pts =cv2.perspectiveTransform(new_pts,np.linalg.inv(warpMatrix).astype(np.float32))
    #apparently the output order has changed in 4.1.1 (possibly earlier from 3.4.3)
    if cv2.__version__<='3.4.4':
        return new_pts[0]
    else:
        return new_pts[:,0,:]


if __name__ ==  "__main__":
    main()
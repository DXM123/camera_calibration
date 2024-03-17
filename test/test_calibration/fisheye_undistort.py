#!/usr/bin/env python3
import cv2
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import sys

# Call this script like this: ./fisheye_undistort.py cam0_20240227_213116.jpg 

# You should replace these 3 lines with the output of "fisheye_calibrate.py"
# R 13 - Cam0
#DIM=(800, 608) # Frame Dimension
#K=np.array([[339.2047580681014, 0.0, 424.2836456711181], [0.0, 335.9774032475118, 318.7268177570001], [0.0, 0.0, 1.0]]) # Matrix
#D=np.array([[0.03660207329421741], [-0.1991110554881065], [0.3176363266667778], [-0.17535892670918304]]) # Dist Coeff

# R 13 - Cam3
DIM=(800, 608)
K=np.array([[325.791702381724, 0.0, 413.06204331054323], [0.0, 324.7871151949905, 275.2780688946987], [0.0, 0.0, 1.0]])
D=np.array([[-0.028782876008750652], [0.04152114093715871], [-0.061831042735305236], [0.03151471884477812]])

def undistort(img_path):
    img = cv2.imread(img_path)
    #h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
import cv2 as cv
import numpy as np

img = cv.imread('1.png')
cv.imshow('img', img)

#step1
x = 97
y = 97

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(x,y)) 

subkernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))   
gradient = cv.morphologyEx(kernel, cv.MORPH_GRADIENT, subkernel)

erosion = cv.erode(img, gradient, iterations=1)
cv.imshow('step1', erosion)

#Step 2
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(x,y)) 
dilation = cv.dilate(erosion, kernel, iterations=1)
cv.imshow('step2', dilation)

#step3
img_or = cv.bitwise_or(img, dilation)
cv.imshow('step3', img_or)

#step4
opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(280,280)) 
img_opening = cv.morphologyEx(img_or, cv.MORPH_OPEN, opening_kernel)

spacer_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
img_spacer = cv.dilate(img_opening, spacer_kernel)

width_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23))
img_width = cv.dilate(img_spacer, width_kernel)

img4 = cv.bitwise_xor(img_spacer, img_width)
cv.imshow('step4', img4)

#step5
img5 = cv.bitwise_and(img, img4)
cv.imshow('step5', img5)

#step6
spacing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(25,25))
img6 = cv.dilate(img5, spacing_kernel)
cv.imshow('step6', img6)

#step7
img7 = cv.subtract(img4, img6)

defect_cue_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(38,38))
img7 = cv.dilate(img7, defect_cue_kernel)
img8 = cv.bitwise_or(img7, img6)

cv.imshow('step7', img8)

cv.waitKey(0)
cv.destroyAllWindows()
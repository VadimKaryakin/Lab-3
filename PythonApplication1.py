import cv2 as cv
import numpy as np

img = cv.imread('1.png', 0)

#kernel = np.zeros((100,100),np.uint8)
x = 95
y = 95

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(x,y)) 
#for i in range(0,x):
#    for j in range(0,y):
#        if kernel[i,j] == 0:
#            kernel[i,j] = 1
#        else:
#            kernel[i,j] = 0

subkernel = np.ones((3,3), np.uint8);
gradient = cv.morphologyEx(kernel, cv.MORPH_GRADIENT, subkernel)

#for i in range(0,10):
#    for j in range(0,):
#        kernel[i,j] = 0
#        kernel[x-i, y-j] = 0
#        kernel[x-i,j] = 0
#        kernel[x,y-i];

#img = cv.bitwise_not(img)
#img = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

erosion = cv.erode(img, gradient, iterations=1)
#erosion = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('se', gradient * 255)
cv.imshow('sample', erosion)
cv.imshow('im', img)
cv.waitKey(0)
cv.destroyAllWindows()

#Step 2
dilation = cv.dilate(erosion, kernel, iterations=1)
cv.imshow('qq', dilation)
cv.waitKey(0)
cv.destroyAllWindows()

#step3

img_or = np.bitwise_or(img, dilation)
cv.imshow('gg', img_or)
cv.waitKey(0)
cv.destroyAllWindows()

#step4
opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(200,200)) 


img_opening = cv.morphologyEx(img_or, cv.MORPH_OPEN, opening_kernel)
cv.imshow('gg', img_opening)
cv.waitKey(0)
cv.destroyAllWindows()
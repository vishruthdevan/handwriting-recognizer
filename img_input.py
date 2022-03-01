from imutils.contours import sort_contours
import numpy as np
import imutils
from imutils import grab_contours
import cv2


image_file = input("Enter path to image: ")
image = cv2.imread(image_file)

greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(grey, (5, 5), 0)

edges = cv2.Canny(blurred, 30, 150)

cv2.imshow("Edges detected: ", edges)

cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)

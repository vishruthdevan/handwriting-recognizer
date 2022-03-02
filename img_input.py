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


cnts = sort_contours(cnts, method="left-to-right")[0]
chars = []
i = 0

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
    roi = greyed[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (tH, tW) = thresh.shape

    if tW > tH:
        thresh = imutils.resize(thresh, width=32)
    else:
        thresh = imutils.resize(thresh, height=32)

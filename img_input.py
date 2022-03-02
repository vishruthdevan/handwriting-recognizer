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

    dX = int(max(0, 32 - tW) / 2.0)
    dY = int(max(0, 32 - tH) / 2.0)
    # pad the image and force 32x32 dimensions
    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    padded = cv2.resize(padded, (32, 32))

    cv2.imwrite(f"results/{str(i)}.png", padded)
    i += 1

    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)
    chars.append((padded, (x, y, w, h)))

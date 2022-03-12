import numpy as np
import imutils
import cv2


def sort_contours(cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][1]))

    sorted_cnts = []
    for c, b in zip(cnts, boundingBoxes):
        appended = False

        if len(sorted_cnts) == 0:
            sorted_cnts.append([(cnts[0], boundingBoxes[0])])
            continue
        for i in range(len(sorted_cnts)):
            if b[1] < sorted_cnts[i][0][1][1] + sorted_cnts[i][0][1][3]:
                sorted_cnts[i].append((c, b))
                appended = True
                break

        if not appended:
            sorted_cnts.append([(c, b)])

    final = []
    print(len(sorted_cnts))
    for i in sorted_cnts:
        s = sorted(i, key=lambda x: x[1][0])
        for j in s:
            print(j[1])
        print("\n\n")
        for j in s:
            final.append(j)
    return final


# image_file = input("Enter path to image: ")
image_file = "images/img3.jpg"
image = cv2.imread(image_file)
greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(greyed, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 150)
cv2.imshow("Edges detected: ", edges)


cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
cnts = sort_contours(cnts)


chars = []
i = 0
_, _, avg_h, avg_w = cv2.boundingRect(cnts[0][0])
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c[0])

    if (w >= avg_w - 60 and w <= avg_w + 60) and (h >= avg_h - 60 and h <= avg_h + 60):
        avg_w = (avg_w + w) / 2
        avg_h = (avg_h + h) / 2

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

        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))

        cv2.imwrite(f"results/{str(i)}.png", padded)
        i += 1

        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        chars.append((padded, (x, y, w, h)))

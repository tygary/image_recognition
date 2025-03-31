# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2


def find_account_number(image_path):
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    docCnt = None
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break


    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))


    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to QR code squares
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    squareCnts = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a QR code square, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            squareCnts.append(c)



    # sort the QR code square contours top-to-bottom, then initialize
    # the total number of correct answers
    squareCnts = contours.sort_contours(squareCnts, method="top-to-bottom")[0]


    account_num_hex = "0x"

    # each row has 4 squares, so loop over the
    # question in batches of 4
    for (q, i) in enumerate(np.arange(0, len(squareCnts), 4)):
        # sort the contours for the current row from
        # left to right
        cnts = contours.sort_contours(squareCnts[i:i + 4])[0]

        binary_values = ""
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # square
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            value = 0 if total < 100 else 1  # threshold to determine if it's a 0 or 1
            binary_values + str(value)
        print("[INFO] Binary values for row {}: {}".format(q + 1, binary_values))
        decimal_value = int(binary_values, 2)  # Convert binary to decimal
        hex_value = hex(decimal_value)[2:]  # Convert decimal to hex, remove "0x" prefix
        account_num_hex + hex_value.upper() 
        # draw the outline of the correct answer on the test
        # cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    account_num = int(account_num_hex, 0)  # Convert hex to integer

    print("[INFO] Account Number", account_num)
    return account_num
    # # grab the test taker
    # score = (correct / 5.0) * 100
    # print("[INFO] score: {:.2f}%".format(score))
    # cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # cv2.imshow("Original", image)
    # cv2.imshow("Exam", paper)
    # cv2.waitKey(0)

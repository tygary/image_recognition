# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import pytesseract

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


# Load TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


"""
TODO Next time:
detect the edge of the paper
segement the paper into distinguishable areas for the account number and text
parse each line of text separately

"""

def find_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(opening, 100, 200)

    text = pytesseract.image_to_string(canny, config='--psm 7 --oem 1 -l eng+osd')
    print("Extracted text: ", text)
    return text

def clean_grid_text_region(roi):
    # Threshold (invert so text is white on black)
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Copy for line removal
    horizontal = binary.copy()
    vertical = binary.copy()

    # Create vertical line kernel (very narrow, tall)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Subtract them from binary
    no_lines = cv2.subtract(binary, vertical_lines)

    no_lines = cv2.bitwise_not(no_lines)

    return no_lines

def find_text_from_contour(gray_image, contour, num_boxes=12):
    # Get bounding rect from contour
    x, y, w, h = cv2.boundingRect(contour)
    # roi = gray_image[y:y+h, x:x+w]
      # Inset the box to skip outer border
    margin = 0.06
    x_in = int(x + (margin * x))
    y_in = int(y + (margin * y))
    w_in = int(w - (0 * margin * w))
    h_in = int(h - (6 * margin * h))

    if w_in <= 0 or h_in <= 0:
        return ""

    roi = gray_image[y_in:y_in + h_in, x_in:x_in + w_in]

    if len(roi.shape) == 2:
        cropped = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    elif cropped.shape[2] == 4:
        cropped = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
    else:
        cropped = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    pil_image = Image.fromarray(cropped)

    # Preprocess and predict
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(inputs.pixel_values, max_length=12,  num_beams=5,  early_stopping=True)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Recognized Text:", text.strip())
    return text.strip()

    # cv2.imwrite("output_name.png", roi)

    # # Optional: preprocess ROI
    # # Threshold using Otsu
    # # cleaned = clean_grid_text_region(roi)
    # _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Optional: Morphological filtering to remove noise
    # # kernel = np.ones((3, 3), np.uint8)
    # # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # # Split into character boxes
    
    # box_width = w // num_boxes
    # line_width = int(box_width * 0.4)
    # characters = []

    # for i in range(num_boxes):
    #     cx = i * box_width
    #     char_img = roi[:, cx:cx+(box_width - line_width)]

    #     char_img = cv2.medianBlur(char_img, 3)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, kernel)

    #     # Resize for better OCR
    #     char_img = cv2.resize(char_img,(0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    #     # Final OCR-ready binarization
    #     _, final_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #     # OCR one character at a time
    #     config = '--psm 10 --oem 1 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    #     text = pytesseract.image_to_string(final_bin, config=config)

    #     characters.append(text.strip())

    # result = ''.join(characters)
    print("Recognized Text:", result)

    # # Optional: Upscale for better OCR results
    # scale_factor = 2
    # resized = cv2.resize(cleaned, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)
    # inverted = cv2.bitwise_not(resized)

    # # OCR
    # config = '--psm 7 --oem 1 -l eng'
    # text = pytesseract.image_to_string(inverted, config=config)

    # print("Extracted text:", text.strip())
    return text.strip()


def find_account_number(area, grayscale_image, paper=None):
    # split into 16 quadrants
    # (4 rows, 4 columns)
    x, y, w, h = cv2.boundingRect(area)

    # trim off the outer edges
    x += 10
    y += 10
    w -= 20
    h -= 20

    num_rows, num_cols = 4, 4
    cell_width = w // num_cols
    cell_height = h // num_rows

    cells = []

    for row in range(num_rows):
        for col in range(num_cols):
            cell_x = x + col * cell_width
            cell_y = y + row * cell_height
            cell = (cell_x, cell_y, cell_width, cell_height)
            cells.append(cell)

    # determine which quadrants are filled
    filled_quadrants = []
    for i, (cx, cy, cw, ch) in enumerate(cells):
        quad = grayscale_image[cy:cy+ch, cx:cx+cw]
    # for i, quad in enumerate(quadrants):
        # Count non-zero pixels in the quadrant
        non_zero_count = cv2.countNonZero(quad)
        percentage_filled = (non_zero_count / (cell_height * cell_width)) * 100
        # print(f"Quadrant {i}: {percentage_filled:.2f}% filled")
        # If the count is above a certain threshold, consider it filled
        if percentage_filled < 20:
            filled_quadrants.append(i)
    
    # convert the filled quadrants to binary
    filled_quadrants_binary = [1 if i in filled_quadrants else 0 for i in range(16)]
    # convert the binary to integer
    filled_quadrants_decimal = int("".join(map(str, filled_quadrants_binary)), 2)
    print(f"Filled quadrants (binary): {filled_quadrants_binary}")
    print(f"Filled quadrants (decimal): {filled_quadrants_decimal}")

    return filled_quadrants_decimal

def parse_form_image(image_path):
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold to binarize the image (black/white)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # or try (7,7)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(closed, 50, 150)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    largest_rect = None
    max_area = 0
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        # loop over the sorted contours
        # print("doc countours: ", cnts)
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    largest_rect = approx
    # print(docCnt)

    docCnt = largest_rect
    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    form = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    contours = imutils.grab_contours(cv2.findContours(form.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
    innerCnts = sorted(contours, key=cv2.contourArea, reverse=True)

    (form_x, form_y, form_w, form_h) = cv2.boundingRect(form)
    account_number_size = form_w // 3
    amount_box_width = form_w // 3
    amount_box_height = form_h // 10
    name_box_width = (form_w * 2) // 3
    name_box_height = form_h // 10
    error_margin = 0.2

    account_number_contours = []
    amount_box_contours = []
    name_box_countours = []
    # loop over the contours
    for c in innerCnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print("counter ", x, " ", y, " ", w, " ", h, " ", ar, " ", len(approx))
        # in order to label the contour as a QR code square, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if len(approx) == 4 and abs(account_number_size - w) < error_margin * account_number_size and abs(account_number_size - h) < error_margin * account_number_size and ar >= 0.9 and ar <= 1.1:
            account_number_contours.append(c)
        elif len(approx) == 4 and abs(amount_box_width - w) < error_margin * amount_box_width and abs(amount_box_height - h) < error_margin * amount_box_height and ar > 4 and ar <= 4.5:
            amount_box_contours.append(c)
        elif len(approx) == 4 and abs(name_box_width - w) < error_margin * name_box_width and abs(name_box_height - h) < error_margin * name_box_height and 8 < ar <= 9:
            name_box_countours.append(c)

    # print("[INFO] countours found: ", len(account_number_contours))
    print("[INFO] account number countours found: ", len(account_number_contours))
    print("[INFO] amount box countours found: ", len(amount_box_contours))
    print("[INFO] name box countours found: ", len(name_box_countours))

    if (len(account_number_contours) == 0 and len(amount_box_contours) == 0 and len(name_box_countours) > 0):
        print("Found Open Account Form")
        name = find_text_from_contour(warped, name_box_countours[0])
        print("Name: ", name)
        return name
    elif (len(account_number_contours) == 1 and len(amount_box_contours) == 0 and len(name_box_countours) == 0):
        print("Found Teller Form")
        account_num = find_account_number(account_number_contours[0], warped)
        print("Account Number: ", account_num)
        return account_num
    elif (len(account_number_contours) == 1 and len(amount_box_contours) >= 1 and len(name_box_countours) == 0):
        print("Found Deposit Form")
        account_num = find_account_number(account_number_contours[0], warped, paper)
        print("Account Number: ", account_num)
        return account_num
        return None
    elif (len(account_number_contours) == 2 and len(amount_box_contours) >= 1 and len(name_box_countours) == 0):
        print("Found Transfer Form")
        if (cv2.boundingRect(account_number_contours[0])[0] < cv2.boundingRect(account_number_contours[1])[0]):
            sorted_account_number_contours = [account_number_contours[0], account_number_contours[1]]
        else:
            sorted_account_number_contours = [account_number_contours[1], account_number_contours[0]]
        from_account_num = find_account_number(sorted_account_number_contours[0], warped)
        to_account_num = find_account_number(sorted_account_number_contours[1], warped)
        print("From Account Number: ", from_account_num, " To Account Number: ", to_account_num)
        return from_account_num, to_account_num
    else:
        print("Found Unknown Form")
        return None

    # account_number = find_account_number(form)



    # # find contours in the thresholded image, then initialize
    # # the list of contours that correspond to QR code squares
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    # innerCnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # print("[INFO] countours found: ", len(innerCnts))

    # squareCnts = []
    # # loop over the contours
    # for c in innerCnts:
    #     # compute the bounding box of the contour, then use the
    #     # bounding box to derive the aspect ratio
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     ar = w / float(h)
    #     print("counter ", x, " ", y, " ", w, " ", h, " ", ar)
    #     # in order to label the contour as a QR code square, region
    #     # should be sufficiently wide, sufficiently tall, and
    #     # have an aspect ratio approximately equal to 1
    #     if w >= 0 and h >= 50 and ar >= 0.5 and ar <= 1.7:
    #         squareCnts.append(c)


    # # sort the QR code square contours top-to-bottom, then initialize
    # # the total number of correct answers
    # squareCnts = contours.sort_contours(squareCnts, method="top-to-bottom")[0]


    # account_num_hex = "0x"

    # # each row has 4 squares, so loop over the
    # # question in batches of 4
    # for (q, i) in enumerate(np.arange(0, len(squareCnts), 4)):
    #     # sort the contours for the current row from
    #     # left to right
    #     cnts = contours.sort_contours(squareCnts[i:i + 4])[0]

    #     binary_values = ""
    #     # loop over the sorted contours
    #     for (j, c) in enumerate(cnts):
    #         # construct a mask that reveals only the current
    #         # square
    #         mask = np.zeros(thresh.shape, dtype="uint8")
    #         cv2.drawContours(mask, [c], -1, 255, -1)
    #         # apply the mask to the thresholded image, then
    #         # count the number of non-zero pixels in the
    #         # bubble area
    #         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    #         total = cv2.countNonZero(mask)
    #         # if the current total has a larger number of total
    #         # non-zero pixels, then we are examining the currently
    #         # bubbled-in answer
    #         value = 0 if total < 100 else 1  # threshold to determine if it's a 0 or 1
    #         binary_values += str(value)
    #     print("[INFO] Binary values for row {}: {}".format(q + 1, binary_values))
    #     decimal_value = int(binary_values, 2)  # Convert binary to decimal
    #     hex_value = hex(decimal_value)[2:]  # Convert decimal to hex, remove "0x" prefix
    #     account_num_hex += hex_value.upper() 
    #     # draw the outline of the correct answer on the test
    #     # cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    # account_num = int(account_num_hex, 0)  # Convert hex to integer

    # print("[INFO] Account Number", account_num)
    # return account_num
    # # # grab the test taker
    # # score = (correct / 5.0) * 100
    # # print("[INFO] score: {:.2f}%".format(score))
    # # cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # # cv2.imshow("Original", image)
    # # cv2.imshow("Exam", paper)
    # # cv2.waitKey(0)


# num = find_account_number("./Withdraw2.jpg")
# print(num)
text = parse_form_image("./OpenAccountReal2.jpg")
# text = parse_form_image("./Teller.png")
# print(text)

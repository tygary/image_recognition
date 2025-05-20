# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import pytesseract
import tensorflow as tf  # Replace tflite_runtime import
import os

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from train_digit_model import load_model_state, save_model_state, fine_tune_model


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

    # Extract the entire grid region
    grid_roi = grayscale_image[y:y+h, x:x+w]
    
    # Enhance contrast
    grid_roi = cv2.equalizeHist(grid_roi)
    
    # Light blur to reduce noise
    grid_roi = cv2.GaussianBlur(grid_roi, (3, 3), 0)
    
    # Apply Otsu's thresholding
    _, binary_grid = cv2.threshold(grid_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a copy of the grayscale image for visualization
    vis_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    
    # Show the thresholded grid in the visualization
    vis_grid = cv2.cvtColor(binary_grid, cv2.COLOR_GRAY2BGR)
    vis_image[y:y+h, x:x+w] = vis_grid

    num_rows, num_cols = 4, 4
    cell_width = w // num_cols
    cell_height = h // num_rows

    # Calculate margins to avoid grid lines
    margin = int(min(cell_width, cell_height) * 0.15)  # 15% margin

    cells = []
    for row in range(num_rows):
        for col in range(num_cols):
            cell_x = col * cell_width + margin
            cell_y = row * cell_height + margin
            cell_w = cell_width - 2 * margin
            cell_h = cell_height - 2 * margin
            cell = (cell_x, cell_y, cell_w, cell_h)
            cells.append(cell)

    # determine which quadrants are filled
    filled_quadrants = []
    for i, (cx, cy, cw, ch) in enumerate(cells):
        # Extract the cell from the already thresholded grid
        binary_quad = binary_grid[cy:cy+ch, cx:cx+cw]
        
        # Count non-zero pixels in the binarized quadrant
        non_zero_count = cv2.countNonZero(binary_quad)
        percentage_filled = (non_zero_count / (cw * ch)) * 100
        # print(f"Quadrant {i}: {percentage_filled:.2f}% filled")
        
        # If more than 40% black pixels (non-zero after inversion), consider it filled
        if percentage_filled > 40:
            filled_quadrants.append(i)
            # Draw a green rectangle around filled cells
            # cv2.rectangle(vis_image, (x+cx-margin, y+cy-margin), 
            #             (x+cx+cw+margin, y+cy+ch+margin), (0, 255, 0), 2)
        # else:
        #     # Draw a red rectangle around empty cells
        #     cv2.rectangle(vis_image, (x+cx-margin, y+cy-margin), 
        #                 (x+cx+cw+margin, y+cy+ch+margin), (0, 0, 255), 2)

    # Save the visualization
    # cv2.imwrite("account_number_visualization.jpg", vis_image)
    
    # convert the filled quadrants to binary
    filled_quadrants_binary = [1 if i in filled_quadrants else 0 for i in range(16)]
    # convert the binary to integer
    filled_quadrants_decimal = int("".join(map(str, filled_quadrants_binary)), 2)
    print(f"Filled quadrants (binary): {filled_quadrants_binary}")
    print(f"Filled quadrants (decimal): {filled_quadrants_decimal}")

    return filled_quadrants_decimal

def detect_contours(form, form_w, form_h, error_margin=0.2):
    """Helper function to detect contours in the form."""
    account_number_size = form_w // 3
    amount_box_width = form_w // 3
    amount_box_height = form_h // 10
    name_box_width = (form_w * 2) // 3
    name_box_height = form_h // 10

    contours = imutils.grab_contours(cv2.findContours(form.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
    innerCnts = sorted(contours, key=cv2.contourArea, reverse=True)

    account_number_contours = []
    amount_box_contours = []
    name_box_countours = []

    for c in innerCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4 and abs(account_number_size - w) < error_margin * account_number_size and abs(account_number_size - h) < error_margin * account_number_size and ar >= 0.9 and ar <= 1.1:
            account_number_contours.append(c)
        elif len(approx) == 4 and abs(amount_box_width - w) < error_margin * amount_box_width and abs(amount_box_height - h) < error_margin * amount_box_height and ar > 4 and ar <= 4.5:
            amount_box_contours.append(c)
        elif len(approx) == 4 and abs(name_box_width - w) < error_margin * name_box_width and abs(name_box_height - h) < error_margin * name_box_height and 8 < ar <= 9:
            name_box_countours.append(c)

    return account_number_contours, amount_box_contours, name_box_countours

def find_number_in_container_ml(gray_image, x, y, w, h, debug_image=None, expected_digits=12576):
    """
    Find a number within a container using TensorFlow model.
    Returns the recognized number or None if no valid number is found.
    If expected_digits is provided, uses it for online learning.
    """
    # Extract ROI
    roi = gray_image[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    # Preprocess ROI
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours of individual digits
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    
    # Filter and sort contours left to right
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c * h_c > 100:  # Filter out noise
            digit_contours.append((x_c, cnt))
    digit_contours.sort(key=lambda x: x[0])  # Sort by x coordinate
    
    if not digit_contours:
        return None

    try:
        # Load model
        model = load_model_state()
        
        # Store all predictions and preprocessed digits
        preprocessed_digits = []
        all_predictions = []
        all_confidences = []
        high_confidence_digits = []
        
        for _, cnt in digit_contours:
            # Extract and preprocess digit
            x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
            digit_roi = roi[y_d:y_d+h_d, x_d:x_d+w_d]
            
            # Pad to make square
            size = max(w_d, h_d)
            padded = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - h_d) // 2
            x_offset = (size - w_d) // 2
            padded[y_offset:y_offset+h_d, x_offset:x_offset+w_d] = digit_roi
            
            # Resize to 28x28 and normalize
            digit_img = cv2.resize(padded, (28, 28))
            preprocessed_digits.append(digit_img)  # Store original image for training
            digit_img = digit_img.astype('float32') / 255.0
            
            # Prepare input tensor
            input_tensor = tf.convert_to_tensor(np.expand_dims(np.expand_dims(digit_img, axis=-1), axis=0))
            
            # Run inference
            predictions = model(input_tensor)
            
            # Get prediction
            prediction = tf.argmax(predictions[0]).numpy()
            confidence = tf.nn.softmax(predictions[0])[prediction].numpy()
            
            # Store all predictions
            all_predictions.append(prediction)
            all_confidences.append(confidence)
            
            # Only use high confidence predictions for output
            if confidence > 0.5:
                high_confidence_digits.append(str(prediction))
                
                # Draw debug visualization
                if debug_image is not None:
                    cv2.putText(debug_image, str(prediction), 
                              (x + x_d, y + y_d - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif debug_image is not None:
                # Draw low confidence predictions in red
                cv2.putText(debug_image, f"{prediction}({confidence:.2f})", 
                          (x + x_d, y + y_d - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Try to fine-tune if we have the expected number of digits
        if len(digit_contours) == len(str(expected_digits)):
            expected_labels = [int(d) for d in str(expected_digits)]
            print(f"Fine-tuning model with digits:")
            for i, (pred, conf) in enumerate(zip(all_predictions, all_confidences)):
                print(f"Position {i}: Predicted {pred} (confidence: {conf:.2f}) -> Expected {expected_labels[i]}")
            
            # Fine-tune the model with all detected digits
            model = fine_tune_model(model, preprocessed_digits, expected_labels)
            save_model_state(model)
        
        # Return high confidence predictions for actual use
        if high_confidence_digits:
            return ''.join(high_confidence_digits)
            
    except Exception as e:
        print(f"Error in ML digit recognition: {str(e)}")
        return None
    
    return None

def parse_form_image(image_path):
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Convert to pure black and white using adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up the image to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find all contours
    cnts = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Sort contours by area in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Look for the form (largest white rectangle with content)
    form_contour = None
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Must be a quadrilateral
        if len(approx) != 4:
            continue
            
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Check aspect ratio (approximately 5:8)
        aspect_ratio = w / float(h)
        if not (1.4 <= aspect_ratio <= 1.8):
            continue
            
        # Extract the ROI
        roi = binary[y:y+h, x:x+w]
        
        # The form should have significant content inside
        # Count the number of black pixels (form content)
        black_pixels = cv2.countNonZero(roi)
        total_pixels = w * h
        black_ratio = black_pixels / float(total_pixels)
        
        # The form should have between 10% and 50% black pixels
        # (too few means it's just white paper, too many means it's not a form)
        if black_ratio < 0.1 or black_ratio > 0.5:
            continue
            
        # Found a good candidate
        form_contour = approx
        break

    if form_contour is None:
        print("No form found")
        return None

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, form_contour.reshape(4, 2))
    warped = four_point_transform(gray, form_contour.reshape(4, 2))

    # Convert warped image to pure black and white
    warped_binary = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    form = cv2.threshold(warped_binary, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    (form_x, form_y, form_w, form_h) = cv2.boundingRect(form)
    
    # Initial contour detection
    account_number_contours, amount_box_contours, name_box_countours = detect_contours(form, form_w, form_h)

    # Check if form is upside down and rotate if needed
    needs_rotation = False
    
    # For account number forms, check if account numbers are in top 20%
    if len(account_number_contours) > 0:
        for contour in account_number_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if y > form_h * 0.2:  # If top of contour is below top 20%
                needs_rotation = True
                break
    
    # For open account form, check if name is in top 20%
    elif len(name_box_countours) > 0:
        for contour in name_box_countours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if y > form_h * 0.2:  # If top of name box is below top 20%
                needs_rotation = True
                break
    
    if needs_rotation:
        print("Form is upside down, rotating...")
        # Rotate the images 180 degrees
        paper = cv2.rotate(paper, cv2.ROTATE_180)
        warped = cv2.rotate(warped, cv2.ROTATE_180)
        form = cv2.rotate(form, cv2.ROTATE_180)
        
        # Re-detect contours after rotation
        account_number_contours, amount_box_contours, name_box_countours = detect_contours(form, form_w, form_h)
    elif len(account_number_contours) > 0:
        print("Form is not upside down")

    # print("[INFO] countours found: ", len(account_number_contours))
    if (len(account_number_contours) > 0):
        print("[INFO] account number countours found: ", len(account_number_contours))
    if (len(amount_box_contours) > 0):
        print("[INFO] amount box countours found: ", len(amount_box_contours))
    if (len(name_box_countours) > 0):
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
        account_num = find_account_number(account_number_contours[0], warped)
        amount = find_number_in_container_ml(warped, x, y, w, h, warped)
        print("Account Number: ", account_num)
        print("Amount: ", amount)
        return account_num, amount
    elif (len(account_number_contours) == 2 and len(amount_box_contours) >= 1 and len(name_box_countours) == 0):
        print("Found Transfer Form")
        if (cv2.boundingRect(account_number_contours[0])[0] < cv2.boundingRect(account_number_contours[1])[0]):
            sorted_account_number_contours = [account_number_contours[0], account_number_contours[1]]
        else:
            sorted_account_number_contours = [account_number_contours[1], account_number_contours[0]]
        from_account_num = find_account_number(sorted_account_number_contours[0], warped)
        to_account_num = find_account_number(sorted_account_number_contours[1], warped)
        amount = find_number_in_container_ml(warped, x, y, w, h, warped)
        print("From Account Number: ", from_account_num, " To Account Number: ", to_account_num)
        print("Amount: ", amount)
        return from_account_num, to_account_num, amount
    else:
        print("Form not recognized")
        return None

# num = find_account_number("./Withdraw2.jpg")
# print(num)
# text = parse_form_image("./OpenAccountReal2.jpg")
# text = parse_form_image("./Teller.png")
# print(text)

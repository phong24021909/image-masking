#!/usr/bin/env python

'''

Usage:
  image_masking.py [<image>]

Keys:
  r     - mask the image
  SPACE - reset the inpainting mask
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import matplotlib.pyplot as plt  # Import matplotlib functionality
import sys  # Enables the passing of arguments
from common import Sketcher
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

my_w = tk.Tk()
my_w.geometry("800x600")  # Size of the window
my_w.title('Thiều Quang Phong-TTCS2022')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Select Image',width=60,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File',
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1)
b3 = tk.Button(my_w, text='Exit',width=20,command=my_w.destroy)
b3.grid(row=10,column=1)
b3.config(bg='#f53d2f')
keys = """        Control Keys:
        r     - mask the image
        SPACE - reset the inpainting mask
        ESC   - end
"""
b4 = tk.Button(my_w, text='Control Keys:\n'
                          '\n'
                          'r     -   mask the image\n'
                          '\n'
                          'SPACE -   reset the inpainting mask\n'
                          '\n'
                          'ESC   -   end',width=40,height=8)
b4.config(bg='#5899ff')
b4.grid(row=3,column=2)
b4. configure(font=("Times New Roman", 13, "italic"))

def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    IMAGE_NAME = filename[:filename.index(".")]
    OUTPUT_IMAGE = IMAGE_NAME + "_output.jpg"
    TABLE_IMAGE = IMAGE_NAME + "_table.jpg"
    b2 =tk.Button(my_w,image=img) # using Button
    b2.grid(row=3,column=1)

    try:
        fn = sys.argv[1]
    except:
        fn = filename

    # Load the image and store into a variable
    image = cv2.imread(cv2.samples.findFile(fn))

    if image is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    # Create an image for sketching the mask
    image_mark = image.copy()
    sketch = Sketcher('Image', [image_mark], lambda: ((255, 255, 255), 255))

    # Sketch a mask
    while True:
        ch = cv2.waitKey()
        if ch == 27:  # ESC - exit
            break
        if ch == ord('r'):  # r - mask the image
            break
        if ch == ord(' '):  # SPACE - reset the inpainting mask
            image_mark[:] = image
            sketch.show()

    # define range of white color in HSV
    lower_white = np.array([0, 0, 255])
    upper_white = np.array([255, 255, 255])

    # Create the mask
    mask = cv2.inRange(image_mark, lower_white, upper_white)

    # Create the inverted mask
    mask_inv = cv2.bitwise_not(mask)

    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract the dimensions of the original image
    rows, cols, channels = image.shape
    image = image[0:rows, 0:cols]

    # Bitwise-OR mask and original image
    colored_portion = cv2.bitwise_or(image, image, mask=mask)
    colored_portion = colored_portion[0:rows, 0:cols]

    # Bitwise-OR inverse mask and grayscale image
    gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
    gray_portion = np.stack((gray_portion,) * 3, axis=-1)

    # Combine the two images
    output = colored_portion + gray_portion

    # Save the image
    cv2.imwrite(OUTPUT_IMAGE, output)

    # Create a table showing input image, mask, and output
    mask = np.stack((mask,) * 3, axis=-1)
    table_of_images = np.concatenate((image, mask, output), axis=1)
    cv2.imwrite(TABLE_IMAGE, table_of_images)

    cv2.imshow('Table of Images', table_of_images)
    cv2.waitKey(0)  # Wait for a keyboard event
    cv2.destroyAllWindows()
my_w.mainloop()  # Keep the window open

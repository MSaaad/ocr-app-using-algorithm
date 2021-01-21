import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    # read in training numbers image
    train_char = cv2.imread("images/training_dataset.png")

    if train_char is None:                          # if image was not read successfully
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    # get grayscale image
    imgGray = cv2.cvtColor(train_char, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(
        imgGray, (5, 5), 0)                        # blur

    # filter image from grayscale to black and white
    threshold_image = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                            255,                                  # make pixels that pass the threshold full white
                                            # use gaussian rather than mean, seems to give better results
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            # invert so foreground will be white, background will be black
                                            cv2.THRESH_BINARY_INV,
                                            # size of a pixel neighborhood used to calculate threshold value
                                            11,
                                            2)                                    # constant subtracted from the mean or weighted mean

    # show threshold image for reference
    cv2.imshow("threshold image", threshold_image)

    # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    threshold_imageCopy = threshold_image.copy()

   # input image, make sure to use a copy since the function will modify this image in the course of finding contours
    # retrieve the outermost contours only
    # compress horizontal, vertical, and diagonal segments and leave only their end points

    Contours, Hierarchy = cv2.findContours(threshold_imageCopy,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # zero rows, enough cols to hold all image data
    flattened_images = np.empty(
        (0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
    classifications = []

    # possible chars we are interested in are digits 0 through 9, put these in list valid_char
    valid_char = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                  ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord(
        'F'), ord('G'), ord('H'), ord('I'), ord('J'),
        ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord(
        'P'), ord('Q'), ord('R'), ord('S'), ord('T'),
        ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for Contour in Contours:
        # if contour is big enough to consider
        if cv2.contourArea(Contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(
                Contour)         # get and break out bounding rect

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(train_char,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW, intY+intH),        # lower right corner
                          (255, 0, 255),                  # color
                          3)                            # thickness

            # crop char out of threshold image
            imgROI = threshold_image[intY:intY+intH, intX:intX+intW]
            # resize image, this will be more consistent for recognition and storage
            imgROIResized = cv2.resize(
                imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # show cropped out char for reference
            cv2.imshow("imgROI", imgROI)
            # show resized image for reference
            cv2.imshow("imgROIResized", imgROIResized)
            # show training numbers image, this will now have red rectangles drawn on it
            cv2.imshow("training_numbers.png", train_char)

            intChar = cv2.waitKey(0)
            if intChar == 27:
                sys.exit()
            elif intChar in valid_char:

                # append classification char to integer list of chars (we will convert to float later before writing to file)
                classifications.append(intChar)

                # flatten image to 1d numpy array so we can write to file later
                single_flattened_image = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # add current flattened impage numpy array to list of flattened image numpy arrays
                flattened_images = np.append(
                    flattened_images, single_flattened_image, 0)

    # convert classifications list of ints to numpy array of floats
    float_classification = np.array(classifications, np.float32)

    # flatten numpy array of floats to 1d so we can write to file later
    npaClassifications = float_classification.reshape(
        (float_classification.size, 1))

    print("\nTraining completed\n")

    # write scanned image metadata to files
    np.savetxt("data-files/classifications.txt", npaClassifications)
    np.savetxt("data-files/flattened_images.txt", flattened_images)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()

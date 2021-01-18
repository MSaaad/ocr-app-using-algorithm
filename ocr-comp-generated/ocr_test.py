import cv2
import numpy as np
import operator
import os
# import argparse
import easygui


MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


# for image upload through arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=False, help="path to testing image")
# args = vars(ap.parse_args())


class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    # calculate bounding rect info
    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA:
            return False        # much better validity
        return True


def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        # read in training classifications
        npaClassifications = np.loadtxt(
            "data-files/classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt(
            "data-files/flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # read in testing numbers image
    # inputTestingImage = cv2.imread("images/4.png")

    # inputTestingImage = cv2.imread(args['image'])

    image_picked = easygui.fileopenbox()

    inputTestingImage = cv2.imread(image_picked)
    if inputTestingImage is None:                           # if image was not read successfully
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    # get grayscale image
    imgGray = cv2.cvtColor(inputTestingImage, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)                    # blur

    # filter image from grayscale to black and white
    # make pixels that pass the threshold full white
    # use gaussian rather than mean, seems to give better results
    # invert so foreground will be white, background will be black
    # size of a pixel neighborhood used to calculate threshold value
    # constant subtracted from the mean or weighted mean

    imgThresh = cv2.adaptiveThreshold(
        imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:
        # instantiate a contour with data object
        contourWithData = ContourWithData()
        # assign contour to contour with data
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(
            contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight(
        )                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(
            contourWithData.npaContour)           # calculate the contour area
        # add contour with data object to list of all contours with data
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():

            # if so, append to valid contour list
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key=operator.attrgetter(
        "intRectX"))         # sort contours from left to right

    # declare final string
    string_output = ""

    for contourWithData in validContoursWithData:

        # draw a border rect around the current char
        cv2.rectangle(inputTestingImage,                                        # draw rectangle on original testing image
                      # upper left corner
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (155, 200, 50),  # border
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        # resize image, recoginition and consitent
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(
            npaROIResized, k=1)     # call KNN function find_nearest

        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # append current char to full string
        string_output = string_output + strCurrentChar
        # string_output = string_output + ' ' + strCurrentChar

    print("\n" + 'Recognized character output:', string_output + "\n")

    # show input image with border boxes drawn around found digits
    cv2.imshow("Input Test Image", inputTestingImage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()

import numpy as np
from scipy.misc.pilutil import imresize
import cv2
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

DIGIT_WIDTH = 20
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10


def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5, 5),
                 cells_per_block=(1, 1),
                 visualize=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

class KNN_MODEL():
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(
            samples, self.k)
        return results.ravel()


class SVM_MODEL():
    def __init__(self, num_feats, C=1, gamma=0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)  
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1, self.features))
        return results[1].ravel()


def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
    final_bounding_rectangles = []

    # find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    for r, hr in zip(bounding_rectangles, hierarchy):
        x, y, w, h = r
        # we are trying to extract ONLY the rectangles with images in it
        # we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        # there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        if ((w*h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            final_bounding_rectangles.append(r)

    return final_bounding_rectangles


def proc_user_img(img_file, model):
    print('Loading %s for digit recognition...' % img_file)
    im = cv2.imread(img_file)
    blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # rectangles of bounding the digits in user image
    digits_rectangles = get_digits(contours, hierarchy)

    for rect in digits_rectangles:
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        im_digit = imgray[y:y+h, x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))

        hog_img_data = pixels_to_hog_20([im_digit])
        pred = model.predict(hog_img_data)
        cv2.putText(im, str(int(pred[0])), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.putText(blank_image, str(
            int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    # plt.imshow(im)
    cv2.imwrite("output-images/original_overlay.png", im)
    cv2.imwrite("output-images/final_digits.png", blank_image)
    cv2.destroyAllWindows()


def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  # row-wise ordering


# this function processes a custom training image
def load_digits_custom(img_file):
    train_data = []
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # rectangles of bounding the digits in user image
    digits_rectangles = get_digits(contours, hierarchy)

    # sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(
        key=lambda x: get_contour_precedence(x, im.shape[1]))

    for index, rect in enumerate(digits_rectangles):
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        im_digit = imgray[y:y+h, x:x+w]
        im_digit = (255-im_digit)

        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class % 10)

        if index > 0 and (index+1) % 10 == 0:
            start_class += 1
    cv2.imwrite("output-images/training_box_overlay.png", im)

    return np.array(train_data), np.array(train_target)

# data preparation
TRAIN_MNIST_IMG = 'digits.png'
TRAIN_USER_IMG = 'custom_train_digits.jpg'
TEST_USER_IMG = 'numbers.jpeg'

# my handwritten dataset
digits, labels = load_digits_custom(TRAIN_USER_IMG)

digits, labels = shuffle(digits, labels, random_state=256)
train_digits_data = pixels_to_hog_20(digits)
X_train, X_test, y_train, y_test = train_test_split(
    train_digits_data, labels, test_size=0.33, random_state=42)

# training and testing
a = int(input("Enter 1 for KNN and 2 for SVM: "))
if a == 1:
    model = KNN_MODEL(k=3)
    model.train(X_train, y_train)
    preds = model.predict(X_test)

    answer = accuracy_score(y_test, preds)
    print('Accuracy with KNN: ', answer*100, '%')

    model = KNN_MODEL(k=4)

    model.train(train_digits_data, labels)

    proc_user_img(TEST_USER_IMG, model)
elif a == 2:
    model = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model.train(X_train, y_train)
    preds = model.predict(X_test)

    answer = accuracy_score(y_test, preds)
    print('Accuracy with SVM: ', answer*100, '%')

    model = SVM_MODEL(num_feats=train_digits_data.shape[1])

    model.train(train_digits_data, labels)
    proc_user_img(TEST_USER_IMG, model)


# import numpy as np
# import cv2

# # Let's take a look at our digits dataset
# image = cv2.imread('digits.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# small = cv2.pyrDown(image)
# cv2.imshow('Digits Image', small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Split the image to 5000 cells, each 20x20 size
# # This gives us a 4-dim array: 50 x 100 x 20 x 20
# cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# # Convert the List data type to Numpy Array of shape (50,100,20,20)
# x = np.array(cells)
# print("The shape of our cells array: " + str(x.shape))

# # Split the full data set into two segments
# # One will be used fro Training the model, the other as a test data set
# train = x[:, :70].reshape(-1, 400).astype(np.float32)  # Size = (3500,400)
# test = x[:, 70:100].reshape(-1, 400).astype(np.float32)  # Size = (1500,400)

# # Create labels for train and test data
# k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# train_labels = np.repeat(k, 350)[:, np.newaxis]
# test_labels = np.repeat(k, 150)[:, np.newaxis]

# # Initiate kNN, train the data, then test it with test data for k=3
# knn = cv2.ml.KNearest_create()
# knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbors, distance = knn.find_nearest(test, k=3)

# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * (100.0 / result.size)
# print("Accuracy is = %.2f" % accuracy + "%")


# def x_cord_contour(contour):
#     # This function take a contour from findContours
#     # it then outputs the x centroid coordinates

#     if cv2.contourArea(contour) > 10:
#         M = cv2.moments(contour)
#         return (int(M['m10']/M['m00']))


# def makeSquare(not_square):
#     # This function takes an image and makes the dimenions square
#     # It adds black pixels as the padding where needed

#     BLACK = [0, 0, 0]
#     img_dim = not_square.shape
#     height = img_dim[0]
#     width = img_dim[1]
#     #print("Height = ", height, "Width = ", width)
#     if (height == width):
#         square = not_square
#         return square
#     else:
#         doublesize = cv2.resize(
#             not_square, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
#         height = height * 2
#         width = width * 2
#         #print("New Height = ", height, "New Width = ", width)
#         if (height > width):
#             pad = (height - width)/2
#             #print("Padding = ", pad)
#             doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad,
#                                                    pad, cv2.BORDER_CONSTANT, value=BLACK)
#         else:
#             pad = (width - height)/2
#             #print("Padding = ", pad)
#             doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0,
#                                                    cv2.BORDER_CONSTANT, value=BLACK)
#     doublesize_square_dim = doublesize_square.shape
#     #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
#     return doublesize_square


# def resize_to_pixel(dimensions, image):
#     # This function then re-sizes an image to the specificied dimenions

#     buffer_pix = 4
#     dimensions = dimensions - buffer_pix
#     squared = image
#     r = float(dimensions) / squared.shape[1]
#     dim = (dimensions, int(squared.shape[0] * r))
#     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     img_dim2 = resized.shape
#     height_r = img_dim2[0]
#     width_r = img_dim2[1]
#     BLACK = [0, 0, 0]
#     if (height_r > width_r):
#         resized = cv2.copyMakeBorder(
#             resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
#     if (height_r < width_r):
#         resized = cv2.copyMakeBorder(
#             resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
#     p = 2
#     ReSizedImg = cv2.copyMakeBorder(
#         resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
#     img_dim = ReSizedImg.shape
#     height = img_dim[0]
#     width = img_dim[1]
#     #print("Padded Height = ", height, "Width = ", width)
#     return ReSizedImg


# image = cv2.imread('numbers.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image", image)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)

# # Blur image then find edges using Canny
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("blurred", blurred)
# cv2.waitKey(0)

# edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

# # Fint Contours
# contours, _ = cv2.findContours(
#     edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Sort out contours left to right by using their x cordinates
# contours = sorted(contours, key=x_cord_contour, reverse=False)

# # Create empty array to store entire number
# full_number = []

# # loop over the contours
# for c in contours:
#     # compute the bounding box for the rectangle
#     (x, y, w, h) = cv2.boundingRect(c)

#     cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#     cv2.imshow("Contours", image)

#     if w >= 5 and h >= 25:
#         roi = blurred[y:y + h, x:x + w]
#         ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
#         squared = makeSquare(roi)
#         final = resize_to_pixel(20, squared)
#         cv2.imshow("final", final)
#         final_array = final.reshape((1, 400))
#         final_array = final_array.astype(np.float32)
#         ret, result, neighbours, dist = knn.find_nearest(final_array, k=1)
#         number = str(int(float(result[0])))
#         full_number.append(number)
#         # draw a rectangle around the digit, the show what the
#         # digit was classified as
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(image, number, (x, y + 155),
#                     cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
#         cv2.imshow("image", image)
#         cv2.waitKey(0)


# cv2.destroyAllWindows()
# print("The number is: " + ''.join(full_number))

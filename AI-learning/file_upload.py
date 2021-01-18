

import easygui
import cv2
# print(easygui.fileopenbox())

x = easygui.fileopenbox()
image = cv2.imread(x)
cv2.imshow('image', image)

cv2.waitKey(0)

cv2.destroyAllWindows()



import cv2
import numpy as np

prototxt_path = "weights/deploy.prototxt.txt"
modelp = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

model = cv2.dnn.readNetFromCaffe( prototxt_path, modelp)
# print(model)

# image = cv2.imread("im1.jpg")
def hsv(img, l, u):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([l,128,128]) # setting lower HSV value
    upper = np.array([u,255,255]) # setting upper HSV value
    mask = cv2.inRange(hsv, lower, upper) # generating mask
    return mask

def filters(imgt):
    img = imgt.copy()
    res = np.zeros(img.shape, np.uint8) # creating blank mask for result
    l = 15 # the lower range of Hue we want
    u = 30 # the upper range of Hue we want
    mask = hsv(img, l, u)
    inv_mask = cv2.bitwise_not(mask) # inverting mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res1 = cv2.bitwise_and(img, img, mask= mask) # region which has to be in color
    res2 = cv2.bitwise_and(gray, gray, mask= inv_mask) # region which has to be in grayscale
    for i in range(3):
        res[:, :, i] = res2 # storing grayscale mask to all three slices
    img = cv2.bitwise_or(res1, res) # joining grayscale and color region
    return img

vid = cv2.VideoCapture(0)

while True:
    _, image = vid.read()
    # print(image.shape)
    h,w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))

    model.setInput(blob)

    out = np.squeeze(model.forward())


    ft_scale = 1.0

    cr = image.copy()

    for i in range(0, out.shape[0]):
        confd = out[i,2]
        if confd > 0.5:
            bx = out[i,3:7]* np.array([w,h,w,h])
            stx,sty,ex,ey = bx.astype(np.int)
            # cr = image[ sty:ey, stx:ex]
            # cc = cv2.Canny(image,50,120, apertureSize=3)
            # cc = cc.reshape(cc.shape[0], cc.shape[1])
            # v , v_color = cv2.pencilSketch(cr, sigma_s=60, sigma_r=0.07, shade_factor=0.1) # inbuilt function to generate pencil sketch in both color and grayscale
            # v , image = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1) # inbuilt function to generate pencil sketch in both color and grayscale
            # image[ sty:ey, stx:ex] = v_color
            # image = cc
            cv2.rectangle( image, ( stx,sty), (ex,ey), color=(255,199,255),thickness=-1)
            cv2.putText(image, f"{confd*100:.2f}%", (stx,sty-5), cv2.FONT_HERSHEY_SIMPLEX, ft_scale, (255,0,0),3)

    cv2.imshow("image", image)
    # cv2.imshow("me_d.jpg", image)
    if cv2.waitKey(1) == ord("q"):
        break
vid.release()
cv2.destroyAllWindows()

# cv2.waitKey(0)

# cv2.imwrite("wow.jpg",image)


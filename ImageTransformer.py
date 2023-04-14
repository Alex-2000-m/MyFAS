import cv2

import os

image_path = 'corpedImages/origin/train/0/'

save_path_hsv = './corpedImages/HSV/train/0/'

save_path_ycrcb = './corpedImages/YCbCr/train/0/'


def brg2hsv_ycrcb(image_path, save_path_hsv, save_path_ycrcb):
    filenames = os.listdir(image_path)

    for filename in filenames:
        examname = filename[:-4]

        type = filename.split('.')[-1]

        img = cv2.imread(image_path + '\\' + filename)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        save_hsv = save_path_hsv + examname + '_HSV' + '.' + type

        save_ycrcb = save_path_ycrcb + examname + '_YCrCb' + '.' + type

        cv2.imwrite(save_hsv, img_hsv)

        cv2.imwrite(save_ycrcb, img_ycrcb)

# def brg2hsv_ycrcb(image_path, save_path_hsv):
#     filenames = os.listdir(image_path)
#
#     for filename in filenames:
#         examname = filename[:-4]
#
#         type = filename.split('.')[-1]
#
#         img = cv2.imread(image_path + '\\' + filename)
#
#         img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#         # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#
#         save_hsv = save_path_hsv + examname + '_HSV' + '.' + type
#
#         # save_ycrcb = save_path_ycrcb + examname + '_YCrCb' + '.' + type
#
#         cv2.imwrite(save_hsv, img_hsv)
#
#         # cv2.imwrite(save_ycrcb, img_ycrcb)


if __name__ == '__main__':
    brg2hsv_ycrcb(image_path, save_path_hsv,save_path_ycrcb)
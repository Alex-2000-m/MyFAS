import cv2
import os
import numpy as np
import math
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import face_recognition  # install from https://github.com/ageitgey/face_recognition

path = './train/1/'
savepath = './corpedImages/origin/train/1/'
# print(path)
for x in os.walk(os.path.dirname(path)):
    number = 1
    for y in x[2]:
        img_name = str(path + y)
        # img_name = './img/Britney_Spears_0004.jpg'

        image_array = cv2.imread(img_name)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        Image.fromarray(image_array)
        face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
        face_landmarks_dict = face_landmarks_list[0]


        # print(face_landmarks_dict, end=" ")

        def visualize_landmark(image_array, landmarks):
            """ plot landmarks on image
            :param image_array: numpy array of a single image
            :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
            :return: plots of images with landmarks on
            """
            origin_img = Image.fromarray(image_array)
            draw = ImageDraw.Draw(origin_img)
            for facial_feature in landmarks.keys():
                draw.point(landmarks[facial_feature])
            imshow(origin_img)


        visualize_landmark(image_array=image_array, landmarks=face_landmarks_dict)


        # plt.show()

        def align_face(image_array, landmarks):
            """ align faces according to eyes position
            :param image_array: numpy array of a single image
            :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
            :return:
            rotated_img:  numpy array of aligned image
            eye_center: tuple of coordinates for eye center
            angle: degrees of rotation
            """
            # get list landmarks of left and right eye
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            # calculate the mean point of landmarks of left and right eye
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")
            # compute the angle between the eye centroids
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            # compute angle between the line of 2 centeroids and the horizontal line
            angle = math.atan2(dy, dx) * 180. / math.pi
            # calculate the center of 2 eyes
            eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                          (left_eye_center[1] + right_eye_center[1]) // 2)
            # at the eye_center, rotate the image by the angle
            # print(eye_center)
            rotate_matrix = cv2.getRotationMatrix2D((int(eye_center[0]), int(eye_center[1])), angle, scale=1)
            rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
            return rotated_img, eye_center, angle


        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=face_landmarks_dict)
        Image.fromarray(np.hstack((image_array, aligned_face)))

        visualize_landmark(image_array=aligned_face, landmarks=face_landmarks_dict)


        # plt.show()

        def rotate(origin, point, angle, row):
            """ rotate coordinates in image coordinate system
            :param origin: tuple of coordinates,the rotation center
            :param point: tuple of coordinates, points to rotate
            :param angle: degrees of rotation
            :param row: row size of the image
            :return: rotated coordinates of point
            """
            x1, y1 = point
            x2, y2 = origin
            y1 = row - y1
            y2 = row - y2
            angle = math.radians(angle)
            x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
            y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
            y = row - y
            return int(x), int(y)


        def rotate_landmarks(landmarks, eye_center, angle, row):
            """ rotate landmarks to fit the aligned face
            :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
            :param eye_center: tuple of coordinates for eye center
            :param angle: degrees of rotation
            :param row: row size of the image
            :return: rotated_landmarks with the same structure with landmarks, but different values
            """
            rotated_landmarks = defaultdict(list)
            for facial_feature in landmarks.keys():
                for landmark in landmarks[facial_feature]:
                    rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
                    rotated_landmarks[facial_feature].append(rotated_landmark)
            return rotated_landmarks


        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                             eye_center=eye_center, angle=angle, row=image_array.shape[0])

        visualize_landmark(image_array=aligned_face, landmarks=rotated_landmarks)


        # plt.show()

        def corp_face(image_array, landmarks, crop_size=256):
            """ crop face according to eye,mouth and chin position
            :param image_array: numpy array of a single image
            :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
            :return:
            cropped_img: numpy array of cropped image
            """

            eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                           np.array(landmarks['right_eye'])])
            eye_center = np.mean(eye_landmark, axis=0).astype("int")
            lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                           np.array(landmarks['bottom_lip'])])
            lip_center = np.mean(lip_landmark, axis=0).astype("int")
            mid_part = lip_center[1] - eye_center[1]
            top = eye_center[1] - mid_part * 30 / 35
            bottom = lip_center[1] + mid_part * 20 / 35

            w = h = bottom - top
            x_min = np.min(landmarks['chin'], axis=0)[0]
            x_max = np.max(landmarks['chin'], axis=0)[0]
            x_center = (x_max - x_min) / 2 + x_min
            left, right = (x_center - w * 30 / (2 * 35), x_center + w * 30 / (2 * 35))

            pil_img = Image.fromarray(image_array)
            left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
            cropped_img = pil_img.crop((left, top, right, bottom))
            # cropped_img = cropped_img.resize((crop_size, crop_size), Image.ANTIALIAS)
            cropped_img = np.array(cropped_img)
            return cropped_img, left, top


        cropped_img, left, top = corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
        img = Image.fromarray(cropped_img)
        savePath = savepath + str(number) + '.jpg'
        img.save(savePath)
        number += 1

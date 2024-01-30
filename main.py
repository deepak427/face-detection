import albumentations as alb
import os
import time
import uuid
import cv2

# Collect Images Using OpenCV

# IMAGES_PATH = os.path.join('data','images')
# number_images = 30

# cap = cv2.VideoCapture(0)
# for imgnum in range(number_images):
#     print('Collecting image {}'.format(imgnum))
#     ret, frame = cap.read()
#     imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# Annotate images with LabelMe


# Review data

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

# images = tf.data.Dataset.list_files('data\\images\\*.jpg')

# def load_image(x):
#     byte_img = tf.io.read_file(x)
#     img = tf.io.decode_jpeg(byte_img)
#     return img

# images = images.map(load_image)

# Visualizing data

# image_generator = images.batch(4).as_numpy_iterator()
# plot_images = image_generator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)
# plt.show()

# Partition unaugmented data

# for folder in ['train', 'test', 'val']:
#     for file in os.listdir(os.path.join('data', folder, 'images')):

#         filename = file.split('.')[0]+'.json'
#         existing_filepath = os.path.join('data', 'labels', filename)
#         if os.path.exists(existing_filepath):
#             new_filepath = os.path.join('data', folder, 'labels', filename)
#             os.replace(existing_filepath, new_filepath)

# Apply Image Augmentation on Images and Labels using Albumentations


augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',
                                                   label_fields=['class_labels']))

img = cv2.imread(os.path.join('data', 'train', 'images',
                 '2a0cb89e-bf63-11ee-83e3-a7b6846881c1.jpg'))
with open(os.path.join('data', 'train', 'labels', '2a0cb89e-bf63-11ee-83e3-a7b6846881c1.json'), 'r') as f:
    label = json.load(f)

coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640, 480, 640, 480]))

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes']
                    [0][:2], [450, 450]).astype(int)),
              tuple(np.multiply(augmented['bboxes']
                    [0][2:], [450, 450]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])

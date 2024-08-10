"""
Perform data augmentation to the images.
Author: @Chiang
"""

import os
import cv2
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt


def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def read_bbox(bbox_path):
  with open(bbox_path, 'r') as file:
    content = file.read()
  bbox = [list(map(float, content.split()))[1:]]
  # elements.append(int(elements.pop(0)))
  return bbox

def save_bbox(output_img_dir, bbox):
  bbox = bbox[0]
  with open(output_img_dir, 'w') as file:
    file.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}')

def augment_img(img, bbox, num_aug, output_img_dir, output_bbox_dir, base_name):
   for i in range(num_aug):
      # Define augmentation pipeline
      transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.Affine(shear=20, p=0.5),
        A.ToGray(p=0.15),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

      # Perform data augmentation
      transformed = transform(image=img, bboxes=bbox, labels=[0])
      transformed_image = transformed['image']
      transformed_bbox = transformed['bboxes']

      # Save augmented image and bounding box
      aug_image_path = os.path.join(output_img_dir, f'{base_name}_{i}.jpg')
      aug_bbox_path = os.path.join(output_bbox_dir, f'{base_name}_{i}.txt')

      cv2.imwrite(aug_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
      save_bbox(aug_bbox_path, transformed_bbox)

def augment_dataset(img_dir, bbox_dir, output_img_dir, output_bbox_dir, num_aug):
  
  os.makedirs(output_img_dir, exist_ok=True)
  os.makedirs(output_bbox_dir, exist_ok=True)

  img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
  for img_file in tqdm(img_files, desc=f"Augmenting {img_dir}"):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    bbox_path = os.path.join(bbox_dir, f'{base_name}.txt')

    img = read_img(img_path)
    bbox = read_bbox(bbox_path)

    augment_img(img, bbox, num_aug, output_img_dir, output_bbox_dir, base_name)


############### Usage ############### 
# img_dir = './imgs/imgs_aug/train/images'
# bbox_dir = './imgs/imgs_aug/train/labels'
# num_aug = 12

# augment_dataset(img_dir, bbox_dir, num_aug)
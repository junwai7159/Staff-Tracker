from utils.k_fold_cv import k_fold_split
from utils.data_augmentation import augment_dataset

# K-Fold Split
save_path = k_fold_split(dataset_path='./imgs/imgs_annotate', ksplit = 5)

##### Data Augmentation #####

# # K-Fold CV 
# for folder in save_path.iterdir():
#   if folder.is_dir():
#     img_dir = folder / 'train/images'
#     bbox_dir = folder / 'train/labels'
#     augment_dataset(img_dir, bbox_dir, img_dir, bbox_dir num_aug=12)

# Whole Dataset
dataset_path = './imgs/imgs_aug'
img_dir = dataset_path + 'train/images'
bbox_dir = dataset_path + 'train/labels'
augment_dataset(img_dir, bbox_dir, img_dir, bbox_dir num_aug=12)
import yaml
import json
import shutil
import datetime
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold

def k_fold_split(dataset_path, ksplit = 5):
  ##### Generating Feature Vectors for Object Detection Dataset #####

  # Retrieve labels of dataset
  dataset_path = Path(dataset_path)
  labels = sorted(dataset_path.rglob('*labels/*.txt'))

  # Extract the indices of the class labels
  yaml_file = dataset_path / 'data.yaml' 
  with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
  cls_idx = sorted(classes.keys())

  # Count the instances of each class-label present in the annotation files
  indx = [l.stem for l in labels]  
  labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
  for label in labels:
    lbl_counter = Counter()

    with open(label, "r") as lf:
      lines = lf.readlines()

    for l in lines:
      # classes for YOLO label uses integer at first position of each line
      lbl_counter[int(l.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

  labels_df = labels_df.infer_objects(copy=False)
  labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`


  ##### K-Fold Dataset Split #####

  ksplit = ksplit
  kf = KFold(n_splits=ksplit, shuffle=True, random_state=42) 
  kfolds = list(kf.split(labels_df))

  # Construct a DataFrame
  folds = [f"split_{n}" for n in range(1, ksplit + 1)]
  folds_df = pd.DataFrame(index=indx, columns=folds)
  for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df.loc[labels_df.iloc[train].index, f"split_{idx}"] = "train"
    folds_df.loc[labels_df.iloc[val].index, f"split_{idx}"] = "val"

  # Calculate the distribution of class labels for each fold as a ratio
  # of the classes present in val to those present in train
  fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
  for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

  # Create directories and data YAML files for each split
  supported_extensions = [".jpg", ".jpeg", ".png"]

  # Initialize an empty list to store image file paths
  images = []

  # Loop through supported extensions and gather image files
  for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

  # Create the necessary directories and dataset YAML files (unchanged)
  save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
  save_path.mkdir(parents=True, exist_ok=True)
  ds_yamls = []

  for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
      yaml.safe_dump(
        {
          "path": split_dir.as_posix(),
          "train": "train",
          "val": "val",
          "names": classes,
        },
        ds_y,
      )

  # Copy images and labels into the respective directory ('train' or 'val') for each split
  for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
      # Destination directory
      img_to_path = save_path / split / k_split / "images"
      lbl_to_path = save_path / split / k_split / "labels"

      # Copy image and label files to new directory (SamefileError if file already exists)
      shutil.copy(image, img_to_path / image.name)
      shutil.copy(label, lbl_to_path / label.name)

  # Save records
  folds_df.to_csv(save_path / "kfold_datasplit.csv")
  fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

  ds_yamls = [str(path) for path in ds_yamls]
  with open(save_path / 'ds_yamls.json', 'w') as file:
    json.dump(ds_yamls, file)

  return save_path
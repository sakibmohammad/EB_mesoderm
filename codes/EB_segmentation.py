import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import keras
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from keras_unet_collection import models, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import random

seed = 42

large_image_stack = tiff.imread("/image_path")
large_mask_stack = tiff.imread("/mask_path")

print(large_image_stack.shape)
print(large_mask_stack.shape)

all_images = [Image.fromarray(large_image_stack[img]) for img in range(large_image_stack.shape[0])]
images = np.stack(all_images, axis=0)
images = np.stack((images,) * 3, axis=-1)  

all_masks = [Image.fromarray(large_mask_stack[msk]) for msk in range(large_mask_stack.shape[0])]
masks = np.stack(all_masks, axis=0)
masks = np.where((masks > 0) & (masks < 255), 0, masks)
masks = np.expand_dims(masks, -1)  

img_height = images.shape[1]
img_width = images.shape[2]
img_channel = images.shape[3]
num_classes = 1
input_shape = (img_height, img_width, img_channel)
batch_size = 4
n_splits = 5  

kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

mean_iou_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Start 5-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(images)):
    print(f"Training fold {fold + 1}/{n_splits}")

    x_train, x_val = images[train_index], images[val_index]
    y_train, y_val = masks[train_index], masks[val_index]
	
    x_train, y_train = x_train /65535.0, y_train/65535.0
    x_val, y_val = x_val/255.0, y_val = 255.0

    model_back_unet = models.att_unet_2d(
        input_size=input_shape,
        filter_num=[64, 128, 256, 512, 1024],
        n_labels=num_classes,
        stack_num_up=2,
        stack_num_down=2,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool=False,
        unpool=False,
        backbone='DenseNet121',
        weights='imagenet',
        freeze_backbone=False,
        freeze_batch_norm=False,
        name='unet_backbone'
    )

    model_back_unet.compile(
        loss=losses.focal_tversky,
        optimizer=Adam(learning_rate=1e-4),
        metrics=[MeanIoU(num_classes=2)]
    )

    model_back_unet.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=1,
        verbose=1
    )

    scores = model_back_unet.evaluate(x_val, y_val, verbose=0)
    print(f"Fold {fold + 1}: Mean IoU: {scores[1]:.4f}")

    y_pred = model_back_unet.predict(x_val)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)
    y_val_bin = (y_val > 0.5).astype(np.uint8)

    precision = precision_score(y_val_bin.flatten(), y_pred_bin.flatten(), zero_division=1)
    recall = recall_score(y_val_bin.flatten(), y_pred_bin.flatten(), zero_division=1)
    f1 = f1_score(y_val_bin.flatten(), y_pred_bin.flatten(), zero_division=1)

    print(f"Fold {fold + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    mean_iou_scores.append(scores[1])
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

print(f"\nAverage Mean IoU: {np.mean(mean_iou_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, DenseNet121, Xception #or some other model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

img_width, img_height = 128, 128
num_classes = 1
seed = 42
cv_splits = 5
batch_size = 8
epochs = 100

initial_lr = 1e-4


def load_images(directory):
    image_paths = glob.glob(os.path.join(directory, '*.tif'))
    images = []
    labels = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize((img_width, img_height))
        image = np.array(image)

        # Normalize the image
        #image = image / 65535.0

        # Append the image to the list
        images.append(image)

        # Extract the label from the directory name
        label = os.path.basename(os.path.dirname(image_path))
        labels.append(label)

    return np.array(images), np.array(labels)

data_dir = '/path_to_data/*'  # Replace with the path to your data directory
x_data, y_data = load_images(data_dir)
x_data = np.stack((x_data,)*3, axis= -1)

datagen_1 = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode='nearest')

datagen_2 = ImageDataGenerator(
                     rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

augmented_images_needed = 400 - len(x_data)

augmented_images = []
augmented_labels = []

while len(augmented_images) < augmented_images_needed:
    for x_batch, y_batch in datagen_1.flow(x_data, y_data, batch_size=batch_size):
        for i in range(x_batch.shape[0]):
            augmented_images.append(x_batch[i])
            augmented_labels.append(y_batch[i])
            if len(augmented_images) >= augmented_images_needed:
                break
        if len(augmented_images) >= augmented_images_needed:
            break

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

x_data = np.concatenate((x_data, augmented_images))
y_data = np.concatenate((y_data, augmented_labels))

skf = StratifiedKFold(n_splits= cv_splits, shuffle=True, random_state = seed)
fold = 1
acc = []
prec = []
rec = []
f1 = []

for train_index, test_index in skf.split(x_data, y_data):
    print(f"Fold: {fold}")

    # Split the data into train and validation sets for this fold
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    x_train = (x_train.astype('float32')) / 65535.
    x_test = (x_test.astype('float32')) / 65535.

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_test = le.transform(y_test)
    y_train = le.transform(y_train)




    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)) #or some other models

    num_layers_to_freeze = int(len(base_model.layers) / 4) 

    for layer in base_model.layers[:num_layers_to_freeze]:
        layer.trainable = False

    # The remaining layers will be trainable
    for layer in base_model.layers[num_layers_to_freeze:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.fit(datagen_2.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(x_train) // batch_size,
              epochs=epochs,
              shuffle = True,
              verbose=1)


    scores = model.evaluate(x_test, y_test, verbose=0)
    acc.append(scores[1] * 100)
    prec.append(scores[2])
    rec.append(scores[3])
    f1.append(2 * scores[2] * scores[3] / (scores[2] + scores[3]))
    fold += 1

for a in acc:
    print("\nAccuracy for this fold is: ", a)

mean_acc = np.mean(acc)
print("\nMean Accuracy for all the folds is: ", mean_acc)

for p in prec:
    print("\nPrecision for this fold is: ", p)

print("\nMean Precision for all the folds is: ", mean_precision)

for r in rec:
    print("\nRecall for this fold is: ", r)

print("\nMean Recall for all the folds is: ", mean_rec)

for f in f1:
    print("\nF1 score for this fold is: ", f)

mean_f1 = np.mean(f1)
print("\nMean F1 score for all the folds is: ", mean_f1)

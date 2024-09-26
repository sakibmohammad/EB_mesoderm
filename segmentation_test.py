import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
import matplotlib


model = load_model("/path_t0_saved_model", compile=False)

test_image_stack = tiff.imread('/tif_image')
test_mask_stack = tiff.imread('/tif_mas')

test_images = []
for img in range(test_image_stack.shape[0]):
    image = test_image_stack[img]
    max_val = np.iinfo(np.uint16).max  
    image = (image.astype('float32')) / max_val
    test_images.append(image)
images = np.array(test_images)
images = np.stack((images,)*3, axis=-1)

test_masks = []
for img in range(test_mask_stack.shape[0]):
    mask = test_mask_stack[img]
    test_masks.append(mask)
masks = np.array(test_masks)
masks = np.expand_dims(masks, -1)

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)

for i in range(images.shape[0]):
    test_img = images[i]
    ground_truth = masks[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

    IOU_keras.update_state(ground_truth[:,:,0], prediction)
    mean_iou = IOU_keras.result().numpy()
    print("Mean IoU for image", i, "=", mean_iou)
    IOU_keras.reset_states()

    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    fig.suptitle('Image ' + str(i) + ', Mean IoU = ' + str(mean_iou), fontweight='bold')
    axs[0].set_title('Testing Image', fontweight='bold')
    axs[0].imshow(test_img, cmap='gray')
    axs[1].set_title('Testing Label', fontweight='bold')
    axs[1].imshow(ground_truth[:,:,0], cmap='gray')
    axs[2].set_title('Prediction on test image', fontweight='bold')
    axs[2].imshow(prediction, cmap='gray')
    plt.show()

## Tool recommendation model for ICSHM2021
## Developed By Kareem Eltouny - University at Buffalo
## Part of: 
## Zhang, X., Eltouny, K., Liang, X., & Behdad, S. (2023)
## Automatic Screw Detection and Tool Recommendation System for Robotic Disassembly. 
## Journal of Manufacturing Science and Engineering, 145(3), 031008.
## 
## 11/15/2021

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import tensorflow as tf
from reset_keras import reset_keras
from efficientnet_v2 import EfficientNetV2S
import numpy as np
import matplotlib.pyplot as plt
import copy
import OsUtils as OsU
import os
#Use this to check if the GPU is configured correctly
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


print("TesnorFlow version: " + tf.__version__)



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--testDir', type=str, default='screws_extracts/test/', help='Test screws crop directory')
parser.add_argument('--cropSize', nargs=2, type=int, default=[128,128], help='Cropped screw image size [width, height]')
parser.add_argument('--batch', type=int, default=5, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1E-5, help='ADAM optimizer learning rate')
parser.add_argument('--modelDir', type=str, default='REVISION_MODELS/model_screws_labeling_4class_aug', help='Directory of saved Tensorflow classification model')
parser.add_argument('-m', '--model', type=str, default='model_screws_labeling_4class_aug', help='name of saved Tensorflow classification model (without ext.)')
parser.add_argument('--output_classes', type=int, default=4, help='model output classes')


args = parser.parse_args()

model_name = args.model
model_dir = args.modelDir

model_filename_save = model_dir +"/" + model_name + ".h5"
training_history = model_dir +"/" + model_name + '_history'
weights_filename_save =  model_dir +"/" + model_name + "_weights.hdf5"
logger_name =  model_dir +"/" + model_name + "_log.csv"

new_size = (args.cropSize[1], args.cropSize[0])

TEST_IMAGES_PATH = args.testDir
batch_size = args.batch
lr = args.lr
output_classes = args.output_classes

test_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
)



validation_generator = test_datagen.flow_from_directory(
    TEST_IMAGES_PATH,
    target_size=new_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=1000,
    shuffle=False
)

input_shape = (args.cropSize[1], args.cropSize[0], 3)

reset_keras()

effnetv2 = EfficientNetV2S(
                    include_top=False, weights='imagenet', input_tensor=None,
                    input_shape=input_shape, pooling=None, classes=3,
                    classifier_activation='softmax', include_preprocessing=False
                )

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),
    effnetv2,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(output_classes, activation='softmax'),
])

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=lr), metrics=["acc"])

   
model.load_weights(weights_filename_save)

# best_model=load_model(model_filename_save)

ev2 = model.evaluate(validation_generator, batch_size=5)

np.savetxt(f'{model_dir}/TEST_loss_and_acc.txt', ev2, fmt='%.4f')


y_softmax = model.predict(validation_generator)
y_pred = np.argmax(y_softmax, axis=1)
y_pred = y_pred.tolist()
np.savetxt(f'{model_dir}/TEST_y_pred.txt', y_pred, fmt='%d')

y_true = []

for i in range(len(validation_generator)):
    t_onehot = validation_generator[i][1]
    t_class = np.argmax(t_onehot, axis=1)
    t_class = t_class.tolist()
    y_true.extend(copy.deepcopy(t_class))

np.savetxt(f'{model_dir}/TEST_y_true.txt', y_true, fmt='%d')

classes = list(validation_generator.class_indices.keys())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
np.savetxt(f'{model_dir}/TEST_cm.txt', cm, fmt='%d')
print(cm)

plt.rcParams.update({'font.size': 20})

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes
                              )
                              
                              
disp = disp.plot(cmap='Blues', values_format='')

plt.savefig(f'{model_dir}/TEST_cm.png', dpi=300)
plt.savefig(f'{model_dir}/TEST_cm.svg', dpi=300)

plt.show()
plt.close()



os.makedirs(f'{model_dir}/misclass', exist_ok=True)
OsU.wipe_dir(f'{model_dir}/misclass')

# for i in range(len(y_pred)):
    # if y_pred[i] != y_true[i]:
        # plt.imshow(validation_generator[i][0])
        # plt.title(f'Classified as {classes[y_pred[i]]}; actual: {classes[y_true[i]]}')
        # plt.savefig(f'{model_dir}/misclass/{i:%d}.png')
        
k = 0

for i in range(len(validation_generator)):
    data = validation_generator[i][0]
    for j in range(len(data)):
        if y_pred[k] != y_true[k]:
            print(f'Screw no. {k} is misclassified as {classes[y_pred[k]]}')
            plt.imshow(data[j].astype('uint8'))
            plt.title(f'Classified as {classes[y_pred[k]]}; actual: {classes[y_true[k]]}')
            plt.savefig(f'{model_dir}/misclass/{k:d}.png')
        k += 1
        






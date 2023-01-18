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
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint 
from efficientnet_v2 import EfficientNetV2S
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
#Use this to check if the GPU is configured correctly
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


print("TesnorFlow version: " + tf.__version__)



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--trainDir', type=str, default='screws_extracts/train/', help='Training screws crop directory')
parser.add_argument('--valDir', type=str, default='screws_extracts/validation/', help='Validation screws crop directory')
parser.add_argument('--cropSize', nargs=2, type=int, default=[128,128], help='Cropped screw image size [width, height]')
parser.add_argument('--batch', type=int, default=5, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1E-5, help='ADAM optimizer learning rate')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
parser.add_argument('--nEpoch', type=int, default=300, help='Maximum number of epochs')
parser.add_argument('--modelDir', type=str, default='model_screws_labeling_4class_aug', help='Directory of saved Tensorflow classification model')
parser.add_argument('-m', '--model', type=str, default='screws_labeling_4class_aug', help='name of saved Tensorflow classification model (without ext.)')


args = parser.parse_args()

model_name = args.model
model_dir = args.modelDir

model_filename_save = model_dir +"/" + model_name + ".h5"
training_history = model_dir +"/" + model_name + '_history'
weights_filename_save =  model_dir +"/" + model_name + "_weights.hdf5"
logger_name =  model_dir +"/" + model_name + "_log.csv"

new_size = (args.cropSize[1], args.cropSize[0])

TRAIN_IMAGES_PATH = args.trainDir
VAL_IMAGES_PATH = args.valDir
batch_size = args.batch
lr = args.lr
patience = args.patience
n_epochs = args.nEpoch


train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
     rotation_range=40,
     width_shift_range=0.3,
     height_shift_range=0.3,
#     shear_range=0.2,
     zoom_range=0.2,
#     horizontal_flip=True,
     fill_mode="nearest",
   #dtype=np.uint8
)

test_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
)


train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=new_size,
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
    seed=0
)
validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=new_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=1000,
    #shuffle=False
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
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax'),
])

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=lr), metrics=["acc"])

model_callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
         ModelCheckpoint(filepath=model_filename_save, monitor='val_loss', save_best_only=True,mode='min'),
         CSVLogger(logger_name, append=True, separator=';')] #,plot_losses - KE 10-6-2021

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=n_epochs,
    callbacks=model_callbacks,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    #max_queue_size=5,
    verbose=1,
    #use_multiprocessing=True,
    #workers=multiprocessing.cpu_count(),
)

# model.save_weights(weights_filename_save)
with open(training_history, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save_weights(weights_filename_save)
   
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(len(loss_train))

plt.figure()
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(f'{model_dir}/loss_vs_epochs.png', dpi=300)
plt.close()


ev2 = model.evaluate(validation_generator, batch_size=5)
np.savetxt(f'{model_dir}/VAL_loss_and_acc.txt', ev2, fmt='%.4f')

validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=new_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=1000,
    shuffle=False
)



y_softmax = model.predict(validation_generator)
y_pred = np.argmax(y_softmax, axis=1)
y_pred = y_pred.tolist()
np.savetxt(f'{model_dir}/VAL_y_pred.txt', y_pred, fmt='%d')

y_true = []

for i in range(len(validation_generator)):
    t_onehot = validation_generator[i][1]
    t_class = np.argmax(t_onehot, axis=1)
    t_class = t_class.tolist()
    y_true.extend(copy.deepcopy(t_class))

np.savetxt(f'{model_dir}/VAL_y_true.txt', y_true, fmt='%d')


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
np.savetxt(f'{model_dir}/VAL_cm.txt', cm, fmt='%d')
print(cm)



from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              )
                              
                              
disp = disp.plot()

plt.savefig(f'{model_dir}/VAL_cm.png', dpi=300)

plt.show()
plt.close()

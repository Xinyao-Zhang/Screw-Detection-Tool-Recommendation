# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:01:35 2021

@author: Seyed Omid Sajedi """

import tensorflow.keras.backend as K
import gc
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow as tf

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect()) # if it's done something you should see a number being outputted
    gc.collect() #
    
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))
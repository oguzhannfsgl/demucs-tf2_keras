import numpy as np
import tensorflow as tf
import os

from cfg import CFG
from model import Demucs
from dataset import MusdbDataset

AUTO = tf.data.experimental.AUTOTUNE


model = Demucs(CFG.NUM_SAMPLES_MODEL, resample=CFG.RESAMPLE, output_sources=len(CFG.OUTPUT_SOURCES), repeat=CFG.BLOCKS,
               use_bn=CFG.USE_BATCH_NORMALIZATION, kernel_initializer=CFG.INITIALIZER)
print("Initialized the model.")

train_filenames = [os.path.join(CFG.TRAIN_FOLDER, folder) for folder in os.listdir(CFG.TRAIN_FOLDER)]
test_filenames = [os.path.join(CFG.TEST_FOLDER, folder) for folder in os.listdir(CFG.TEST_FOLDER)]
train_dataset = MusdbDataset(train_filenames, CFG.BATCH_SIZE, augment=CFG.AUGMENT)
val_dataset = MusdbDataset(test_filenames, CFG.VAL_BATCH_SIZE, augment=False)
print("Created the dataset.")


def l1_loss(y_true, y_pred):
    return y_pred
    
model.compile(loss=l1_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LR), metrics=[l1_loss])


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                patience=CFG.LR_REDUCER_PATIENCE,
                                                verbose=1)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=CFG.EARLY_STOPPER_PATIENCE,
                                                verbose=1,
                                                restore_best_weights=True)
checkpoint_path = os.path.join(os.getcwd(), "weights", "demucs-epoch_{epoch}-val_loss_{val_loss}")
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                monitor="val_loss",
                                                verbose=0,
                                                save_best_only=True,
                                                save_weights_only=True)

callbacks = [lr_reducer, early_stopper, checkpoint]

model.fit(train_dataset, validation_data=val_dataset, epochs=CFG.EPOCHS, callbacks=callbacks)

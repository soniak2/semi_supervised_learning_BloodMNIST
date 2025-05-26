import tensorflow as tf
import datetime

def get_callbacks(folder_name, run_name):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.1, 
                                  patience=5, 
                                  min_lr=1e-6,
                                  verbose = False
                                  )

    log_dir = f"logs/{folder_name}/{run_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                    histogram_freq=1,
                                    write_graph=True,
                                    #write_images=False,
                                    #write_steps_per_second=False,
                                    update_freq='epoch',
                                    #profile_batch=0,
                                    #embeddings_freq=0,
                                    #embeddings_metadata=None
                                )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    #min_delta=0,
                                                    patience=10,
                                                    verbose=0,
                                                    #mode='auto',
                                                    baseline=None,
                                                    restore_best_weights=True,
                                                    start_from_epoch=40
                                                )
    
    return [reduce_lr, tensorboard, early_stop]

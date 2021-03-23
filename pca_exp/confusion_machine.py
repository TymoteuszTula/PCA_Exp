# confusion_machine.py

import tensorflow as tf
import numpy as np

class ConfusionMachine:

    def __init__(self, neuron_vector, activation_vector):
        
        self.conf_model = tf.keras.models.Sequential()
        self.conf_model.add(tf.keras.Input(shape=(neuron_vector[0],)))

        for i in range(1, len(neuron_vector)):
            self.conf_model.add(tf.keras.layers.Dense(neuron_vector[i], 
                                    activation=activation_vector[i-1]))

        self.conf_model.add(tf.keras.layers.Dense(2, 
                                  activation=activation_vector[-1]))
        
        self.loss_conf = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        self.conf_model.save_weights('weights.h5')

    def perform_confusion_on_pcs(self, pca_machine, param, 
                                    param_range, res_idx=0, up_to=5):

        acc = []
        
        pc_scores = pca_machine.pc_scores[res_idx][:up_to,:].T
        pc_size = pc_scores.shape[0]

        for N_c in param_range:
            idx_less = np.argwhere(param < N_c)[:,0]
            labels = np.zeros((pc_size,))
            labels[idx_less] = np.ones((idx_less.size,))

            self.conf_model.compile(optimizer='adam',
                                    loss=self.loss_conf,
                                    metrics=['accuracy'])
            hist = self.conf_model.fit(pc_scores, labels, epochs=200)

            acc.append(hist.history.get('accuracy')[-1]) 

            self.conf_model.load_weights('weights.h5')

        return acc





            
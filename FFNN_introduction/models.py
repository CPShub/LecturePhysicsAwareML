"""
Lecture "Physics-aware Machine Learning" (SoSe 23/24)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Jasper O. Schommartz
         
04/2024
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
from tensorflow.keras import layers
import datetime
now = datetime.datetime.now


# %%   
"""
MLP: custom trainable layer

"""

class MLP(layers.Layer):
    def __init__(self, units, activation, non_neg):
        super().__init__()
        
        self.ls = []
        
        for (u, a, n) in zip(units, activation, non_neg):
            
            if n:
                kernel_constraint = tf.keras.constraints.non_neg()
                
            else:
                kernel_constraint = None
                
            self.ls += [layers.Dense(u, a, kernel_constraint=kernel_constraint)]      

         
    def call(self, x):     
        
      
        for l in self.ls:
            x = l(x)
        return x


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[1])
    # define which (custom) layers the model uses
    ys = MLP(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model
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
import numpy as np


# %%   
"""
Generate data 

"""


def get_data(case):
    
    if case == 'bathtub':
        
        xs = np.linspace(0,1,500)
        ys = np.concatenate([np.square((xs[0:150]*10.0-3.0))*0.1, np.zeros(200), \
                             np.square((xs[350:500]*10-7.0))*0.1])
        
    elif case == 'curve':
    
        xs = np.linspace(0,1,500)
        ys = np.tanh(-3 + (6*xs)) / 2.0 + xs / 2.0  
        
    elif case == 'double_curve':
    
        xs = np.linspace(0,1,500)
        ys = np.sin(2.0 * np.pi * xs) / 3.0 + xs
        

    xs = tf.expand_dims(xs, axis = 1)
    ys = tf.expand_dims(ys, axis = 1)

    xs_c = np.concatenate([xs[0:260,:], xs[330:480,:]])
    ys_c =np.concatenate([ys[0:260,:], ys[330:480,:]])
        

        
    return xs, ys, xs_c, ys_c



def get_data_with_gradients(case):
    
    if case == 'bathtub':
        
        xs = np.linspace(0,1,500)
        ys = np.concatenate([np.square((xs[0:150]*10.0-3.0))*0.1, np.zeros(200), \
                             np.square((xs[350:500]*10-7.0))*0.1])
        dys = np.concatenate([(xs[0:150]*10.0-3.0)*2.0, np.zeros(200), \
                             (xs[350:500]*10-7.0)*2.0])
        
    elif case == 'curve':
    
        xs = np.linspace(0,1,500)
        ys = np.tanh(-3 + (6*xs)) / 2.0 + xs / 2.0  
        dys = 3.5 - 3.0 * np.square(np.tanh(-3 + (6*xs)))
        
    elif case == 'double_curve':
    
        xs = np.linspace(0,1,500)
        ys = np.sin(2.0 * np.pi * xs) / 3.0 + xs
        dys = 2.0/3.0*np.pi*np.cos(2.0*np.pi*xs) + 1.0
        

    xs = tf.expand_dims(xs, axis = 1)
    ys = tf.expand_dims(ys, axis = 1)
    dys = tf.expand_dims(dys, axis = 1)

    xs_c = np.concatenate([xs[0:260,:], xs[330:480,:]])
    ys_c =np.concatenate([ys[0:260,:], ys[330:480,:]])
    dys_c =np.concatenate([dys[0:260,:], dys[330:480,:]])
        

        
    return xs, ys, dys, xs_c, ys_c, dys_c





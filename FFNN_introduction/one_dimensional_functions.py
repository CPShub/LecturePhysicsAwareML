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
import datetime
now = datetime.datetime.now

# %% Own modules
import PAML_test.FFNN_introduction.data as ld
import PAML_test.FFNN_introduction.models as lm
import PAML_test.FFNN_introduction.plots as lp



def one_dimensional_functions(units, activation, non_neg, data, epochs):
    #   units: number of nodes in each hidden layer
    #   acts: activation function in each hidden layer
    #   non_neg: restrict the weights in different layers to be non-negative


    #   load model

    model = lm.main(units=units, activation=activation, non_neg=non_neg)


    #   load data

    xs, ys, xs_c, ys_c = ld.get_data(data)


    #   calibrate model

    t1 = now()
    print(t1)

    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
    h = model.fit([xs_c], [ys_c], epochs = epochs,  verbose = 2)

    t2 = now()
    print('it took', t2 - t1, '(sec) to calibrate the model')

    lp.plot_loss(h)
    
    #   evaluate model

    lp.plot_data_model(xs, ys, xs_c, ys_c, model, data, 4)






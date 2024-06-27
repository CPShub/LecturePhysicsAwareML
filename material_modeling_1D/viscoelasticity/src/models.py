import tensorflow as tf
from tensorflow.keras import layers

def make_layer(m_type, params, **kwargs):
    """ Factory method that calls and returns layer object """
    cf = {
        'Maxwell': MaxwellAnalyticalCell,
        'Naive': NaiveRNNCell,
        'GSM': GSMCell
        }
    class_obj = cf.get(m_type, None)
    if class_obj:
        return class_obj(params, **kwargs)
    raise ValueError('Unknown class type')

class MLP(layers.Layer):
    ''' A feed-forward neural network '''
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
    

class MaxwellAnalyticalCell(layers.Layer):
    ''' Fully analytical Maxwell model with explicit Euler time integration '''
    def __init__(self, params, **kwargs):
        super().__init__()
        self.state_size = [[1]]
        self.output_size = [[1]]
     
        self.E_infty = params['E_infty']
        self.E = params['E']
        self.eta = params['eta']
        
    def call(self, inputs, states):
        '''   states are the internal variables
           n: current time step, N: old time step '''
        eps_N = inputs[0]
        hs_N = inputs[1]
        gamma_N = states[0]
        
        # compute stress
        sig_N = self.E_infty * eps_N + self.E * (eps_N - gamma_N)

        # integrate internal variable for next time step
        gamma_n = gamma_N + hs_N * self.E/self.eta * (eps_N - gamma_N)
                
        return sig_N, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        ''' Define the initial values of the internal variables. '''     
        return [tf.zeros([batch_size, 1])]
    
class NaiveRNNCell(layers.Layer):
    ''' Naive RNN model with fully NN-based material model and time integration '''
    def __init__(self, params, **kwargs):
        super().__init__()
        self.state_size = [[1]]
        self.output_size = [[1]]
     
        self.mlp = MLP(**kwargs)
        
    def call(self, inputs, states):
        ''' states are the internal variables
            n: current time step, N: old time step '''

        # concatenate NN inputs   
        eps_N = inputs[0]
        hs_N = inputs[1]
        gamma_N = states[0]
        x = tf.concat([eps_N, hs_N, gamma_N], axis = 1)

        # Evaluate NN 
        x = self.mlp(x)
         
        sig_N = x[:, 0:1]
        gamma_n = x[:, 1:2]
        
        return sig_N, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        ''' Define the initial values of the internal variables.'''  
        return [tf.zeros([batch_size, 1])]
    

class GSMCell(layers.Layer):
    ''' GSM cell '''
    def __init__(self, params, **kwargs):
        super().__init__()
        self.state_size = [[1]]
        self.output_size = [[1]]
        self.eta = params['eta']

        self.mlp = MLP(**kwargs)
        
    def call(self, inputs, states):
        '''  states are the internal variables
        n: current time step, N: old time step '''
                
        eps_N = inputs[0]
        hs_N = inputs[1]
        gamma_N = states[0]
        x = tf.concat([eps_N, gamma_N], axis = 1)
        
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.mlp(x)
        dedgamma = g.gradient(y, x)

        sig_N = dedgamma[:, 0, tf.newaxis]
        gamma_n = gamma_N - hs_N * self.eta**(-1) * dedgamma[:, 1, tf.newaxis]
                
        return sig_N , [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        ''' Define initial values of the internal variables. '''
        return [tf.zeros([batch_size, 1])]

def build(m_type, params, **kwargs):
    ''' Build the model. '''
    # eps = tf.keras.Input(shape=[None, 1])
    # hs = tf.keras.Input(shape=[None, 1])
    eps = tf.keras.Input(shape=(1,))
    hs = tf.keras.Input(shape=(1,))

    print(eps)
    print(hs)
        
    cell = make_layer(m_type, params, **kwargs)
    print(cell)
    rnn = layers.RNN(cell, return_sequences=True, return_state=False)
    print(rnn)
    sigs = rnn((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    return model
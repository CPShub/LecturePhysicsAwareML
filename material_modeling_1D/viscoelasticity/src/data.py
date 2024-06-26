"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2023
"""


# %%   
"""
Import modules

"""

import numpy as np
import tensorflow as tf

tf_dt = 'float32'


def harmonic_data(E_infty, E, eta, n, omega, A):
    
    """
    Solution of the generalized Maxwell model, using the explicit Euler scheme.
    
    n: total amount of time steps
    periods: number of periods
    amplitude: amplitude of the oscillation

    
    E_infty: stiffness of equilibrium spring
    E: stiffness of non-equilibrium spring
    eta: viscosity of damper

    """
    
    
    t = np.linspace(0, 2*np.pi,n)

    eps = A * np.sin(omega * t)
    eps_dot = A * omega * np.cos(omega * t)
    sig = np.zeros_like(eps)
    gamma = np.zeros_like(eps)
    dts = np.zeros_like(eps)

    
    for i in range(len(t)-1):
        
        dts[i+1] = t[i+1] - t[i]
        
        gamma[i+1] = gamma[i] + dts[i+1] * E / eta * ( eps[i] - gamma[i] )
        
        sig[i+1] = E_infty * eps[i+1] + E * ( eps[i+1] - gamma[i+1] )
        
    
       
    eps = tf.constant(eps, dtype=tf_dt)
    eps = tf.expand_dims(eps, axis = 0)
    eps = tf.expand_dims(eps, axis = 2)
    
        
    eps_dot = tf.constant(eps_dot, dtype=tf_dt)
    eps_dot = tf.expand_dims(eps_dot, axis = 0)
    eps_dot = tf.expand_dims(eps_dot, axis = 2)
    
    sig = tf.constant(sig, dtype=tf_dt)
    sig = tf.expand_dims(sig, axis = 0)
    sig = tf.expand_dims(sig, axis = 2)
    
    dts = tf.constant(dts, dtype=tf_dt)
    dts = tf.expand_dims(dts, axis = 0)
    dts = tf.expand_dims(dts, axis = 2)
        
    return eps, eps_dot, sig, dts


def relaxation_data(E_infty, E, eta, n, omega, A):
    
    t = np.linspace(0, 2*np.pi,n)
    
    n1 = int(np.round(n / 4.0 / omega))

    eps = A * np.sin(omega * t[0:n1])
    eps_dot = A * omega * np.cos(omega * t[0:n1])
    
    eps = np.concatenate([eps, A * np.sin(omega * t[n1])*np.ones(n-n1)])
    eps_dot = np.concatenate([eps_dot, np.cos(omega * t[n1])*np.zeros(n-n1)])

    sig = np.zeros_like(eps)
    gamma = np.zeros_like(eps)
    dts = np.zeros_like(eps)
    
    
    for i in range(len(t)-1):
        
        dts[i] = t[i+1] - t[i]
        
        gamma[i+1] = gamma[i] + dts[i] * E / eta * ( eps[i] - gamma[i] )
        
        sig[i+1] = E_infty * eps[i+1] + E * ( eps[i+1] - gamma[i+1] )
        
    
       
    eps = tf.constant(eps, dtype=tf_dt)
    eps = tf.expand_dims(eps, axis = 0)
    eps = tf.expand_dims(eps, axis = 2)
    
        
    eps_dot = tf.constant(eps_dot, dtype=tf_dt)
    eps_dot = tf.expand_dims(eps_dot, axis = 0)
    eps_dot = tf.expand_dims(eps_dot, axis = 2)
    
    sig = tf.constant(sig, dtype=tf_dt)
    sig = tf.expand_dims(sig, axis = 0)
    sig = tf.expand_dims(sig, axis = 2)
    
    dts = tf.constant(dts, dtype=tf_dt)
    dts = tf.expand_dims(dts, axis = 0)
    dts = tf.expand_dims(dts, axis = 2)
        
    return eps, eps_dot, sig, dts


def generate_data_harmonic(E_infty, E, eta, n, omegas, As):
    
    eps = []
    eps_dot = []
    sig = []
    dts = []
    
    
    for i in range(len(omegas)):
        
        eps2, eps_dot2, sig2, dts2 = harmonic_data(E_infty, E, eta, \
                                                n, omegas[i], As[i])
            
        eps.append(eps2)
        eps_dot.append(eps_dot2)
        sig.append(sig2)
        dts.append(dts2)
    
    eps = tf.concat(eps, 0)
    eps_dot = tf.concat(eps_dot, 0)
    sig = tf.concat(sig, 0)
    dts = tf.concat(dts, 0)
    
    return eps, eps_dot, sig, dts


def generate_data_relaxation(E_infty, E, eta, n, omegas, As):
    
    eps = []
    eps_dot = []
    sig = []
    dts = []
    
    
    for i in range(len(omegas)):
        
        eps2, eps_dot2, sig2, dts2 = relaxation_data(E_infty, E, eta, \
                                                n, omegas[i], As[i])
            
        eps.append(eps2)
        eps_dot.append(eps_dot2)
        sig.append(sig2)
        dts.append(dts2)
    
    eps = tf.concat(eps, 0)
    eps_dot = tf.concat(eps_dot, 0)
    sig = tf.concat(sig, 0)
    dts = tf.concat(dts, 0)
    
    return eps, eps_dot, sig, dts
            



        







import numpy as np
import math
import scipy
import time
from matplotlib import pyplot as plt

def r_func():
    """
    position function
    """

def rdot_func():
    """
    velocity function
    """
def contraints(des_vars, params):
    t = des_vars[0]
    Tc = des_vars[1]
    Gamma = des_vars[2]

    #constraint 5
    #x in design space, t in range(0, tf) --> trivial

    #constraint 7
    # initial mass --> trivial

    #constraint 8
    # intial position and velocity --> trivial

    #constraint 9
    # geometry between final position and velocity --> trivial

    #constraint 17
    xdot = A_func(omega)*x_func(t) + B*(g + Tc/m)
    mdot = -params["alpha"] * Gamma

    #constraint 18
    res1 = Gamma - np.linalg.norm(Tc)
    res2 = params["rho2"] - Gamma
    res3 = params["rho1"]


    return

def convexified(des_vars, params):

    """ Problem 3: convex relaxed min landing error problem """
    tf = des_vars[0]
    Tc = des_vars[1]
    Gamma = des_vars[2]

    f3 = np.linalg.norm(E*r_func(tf) - q)

    # add gradient
    g3 = np.zeros(2)

    """problem 4: convex relaxed min fuel problem"""

    f4 = np.sum(Gamma, axis=0)

    # add gradient
    g4 = np.zeros(2)

    f = f3 + f4
    g = g3 + g4

    return f, g

if __name__ == '__main__':

    ###initialization
    x


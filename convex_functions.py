import numpy as np
import math
import scipy
from matplotlib import pyplot as plt

def plot_trajectory(time, x):
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10))
    axs1[0].plot(time, x[:, 0], label='x')
    axs1[0].set_ylabel('x')
    axs1[0].set_title('Figure 1: x, y, z vs Time')
    axs1[0].grid()
    axs1[1].plot(time, x[:, 1], label='y')
    axs1[1].set_ylabel('y')
    axs1[1].grid()
    axs1[2].plot(time, x[:, 2], label='z')
    axs1[2].set_ylabel('z')
    axs1[2].set_xlabel('Time')
    axs1[2].grid()

    # Create Figure 2: xdot, ydot, zdot vs time
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 10))
    axs2[0].plot(time, x[:, 3], label='xdot')
    axs2[0].set_ylabel('xdot')
    axs2[0].set_title('Figure 2: xdot, ydot, zdot vs Time')
    axs2[0].grid()
    axs2[1].plot(time, x[:, 4], label='ydot')
    axs2[1].set_ylabel('ydot')
    axs2[1].grid()
    axs2[2].plot(time, x[:, 5], label='zdot')
    axs2[2].set_ylabel('zdot')
    axs2[2].set_xlabel('Time')
    axs2[2].grid()

    plt.tight_layout()
    plt.show()
    plt.close()

def S_func(omega):
    # eqn2
    S = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    return S


def A_func(omega):
    A = np.zeros((6,6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -S_func(omega) ** 2
    A[3:6, 3:6] = -2 * S_func(omega)
    #A = np.array([[np.zeros((3, 3)), np.eye(3)], [-S_func(omega) ** 2, -2 * S_func(omega)]])  # 6x6
    return A

def rdot_func(z, t, Tc, params):
    """
    dynamics function: lander modeled as a lumped parameter mass with Tc for control
    omega = np.ones((3,)) #vector of planets constant angular velocity
    x: 6x1
    """
    x = z[0:6]
    m = z[3]

    omega = params["omega"]

    g = np.ones((3,))  # constant gravity vector
    S = S_func(omega)

    A = A_func(omega)  # 6x6
    B = np.zeros((6,3))
    B[3:6,:] = np.eye(3)

    xdot = np.dot(A, x) + np.dot(B, g + Tc / m)  # 6x1
    mdot = params["alpha"] * np.linalg.norm(Tc)
    zdot = np.append(xdot, mdot)

    return zdot


def contraints(tf, Tc, Gamma, omega, params):
    # constraint 5
    # x in design space, t in range(0, tf) --> trivial

    # constraint 7
    # initial mass --> trivial

    # constraint 8
    # intial position and velocity --> trivial

    # constraint 9
    # geometry between final position and velocity --> trivial

    # constraint 17
    xdot = A_func(omega) * x_func(tf) + B * (g + Tc / m)
    mdot = -params["alpha"] * Gamma

    # constraint 18
    res1 = Gamma - np.linalg.norm(Tc)
    res2 = params["rho2"] - Gamma
    res3 = Gamma - params["rho1"]

    # constraint 19
    n = np.array([1, 0, 0])  # normal direction
    res4 = np.dot(n.T, Tc) - math.cos(theta) * Gamma

    # constraint 20
    # second term is the obj function for problem 3
    # d_star_p3 is final optimal position from problem 3

    res5 = np.linalg.norm(d_star_p3 - q) - np.linalg.norm(E * r_func(tf) - q)

    return


def problem3(des_vars, params):
    """ Problem 3: convex relaxed min landing error problem """
    tf = des_vars[0]
    Tc = des_vars[1]
    Gamma = des_vars[2]

    f3 = np.linalg.norm(E * r_func(tf) - q)

    # add gradient
    g3 = np.zeros(2)

    return f3, g3

def problem4(des_vars, params):
    """problem 4: convex relaxed min fuel problem"""

    f4 = np.sum(Gamma, axis=0)

    # add gradient
    g4 = np.zeros(2)

    return f4, g4


if __name__ == '__main__':
    ### constants
    Tmax = 24000  # N
    params = {
        "rho1": 0.2 * Tmax,  # thrust limit for throttle level 20%
        "rho2": 0.8 * Tmax,
        "alpha": 5e-4,  # s/m
        "omega": np.array([1, 1, 1]),  # PLACEHOLDER
    }

    ### initialization
    m0 = 2000  # kg
    mf = 300  # kg
    r0 = np.array([2400, 450, -330])  # m
    rdot0 = np.array([-10, -40, 10])  # m/s
    x0 = np.zeros((6,))
    x0[0:3] = r0
    x0[3:6] = rdot0
    q = 0  # m, target landing site

    # discretize time

    # propogate dynamics
    z0 = np.append(x0, m0)
    tarray = np.linspace(0, 10, 1000)
    Tc = 0.5 * Tmax  # assuming constant thrust over this short timer
    z = scipy.integrate.odeint(rdot_func, z0, tarray, args=(Tc, params))
    x = z[:, 0:6]
    m = z[:, 6]
    plot_trajectory(tarray, x)

    # optimize problem 3, 4
    # solve problem3 for dp3

    # solve problem4 with dp3


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
    A[3:6, 0:3] = -np.dot(S_func(omega),S_func(omega))
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

def discretize_problem(N, tf, params):
    """
    1. sets up the time grid, solves for control inputs at each time step
    2. and propagates the dynamics using those control inputs.
    """
    dt = tf / N

    # Define the time grid
    tgrid = np.linspace(0, tf, N + 1)

    # Initialize an array to store control inputs at each time step
    Tc_array = np.zeros((N, 3))  # Assuming 3 control inputs

    # Solve for optimal control inputs at each time step
    for i in range(N):
        Tc_i = Tc_array[i]  # Get control input at time step i

        # Call your optimizer (modify it to work for a specific time step)
        # Example: You may call scipy.optimize.minimize here for each time step

        # Store the obtained optimal control input for the specific time step
        Tc_array[i] = obtained_Tc_i  # Replace obtained_Tc_i with the actual result

    # Propagate dynamics using the obtained control inputs
    z0 = np.append(x0, m0)
    x_array = np.zeros((N + 1, 6))  # Store states at each time step
    x_array[0] = x0

    for i in range(N):
        Tc = Tc_array[i]
        tspan = [tgrid[i], tgrid[i + 1]]  # Time span for this step

        # Integrate dynamics for this time step
        z_step = scipy.integrate.odeint(rdot_func, x_array[i], tspan, args=(Tc, params))

        # Store the resulting state for this time step
        x_array[i + 1] = z_step[-1]

    return tgrid, x_array, Tc_array

def constraints_func(Tc, params):

    """ setup """
    n = np.array([1, 0, 0])  # normal direction
    Vmax = params["Vmax"]
    error_margin = params["error_margin"]
    Gamma = np.linalg.norm(Tc)
    q = params["q"]
    g = np.array([-3.71, 0, 0])  # constant gravity vector
    E = np.zeros((2, 3))
    E[0, :] = np.array([0, 1, 0])
    E[1, :] = np.array([0, 0, 1])
    e1 = np.array([1, 0, 0])  # axial direction
    n = np.array([1, 0, 0])  # normal direction
    gamma = math.pi/4 #glide slope angle
    theta = math.pi/4 #thrust pointing constraint
    c = e1 / math.tan(gamma)  # glide slope direction
    error_margin = 10  # m, upate for pareto-front study

    """ residuals of the constraints """
    #res5 = np.linalg.norm(d_star_p3 - q) - np.linalg.norm(E * r_func(tf) - q)
    res5a = Vmax - np.linalg.norm(np.array[v_x, v_y, v_z])
    res5b = np.linalg.norm(E * r_func(tf) - c.T * (np.array[x, y, z] - r_func(tf)))  # HOW DO WE KNOW r_tf???
    res7b = m - m0 - mf
    res9a = np.dot(e1, np.array[x, y, z])
    res9b = np.array([v_x, v_y, v_z])  # constraint: rdot(tf) = 0
    res17a = A_func(omega) * X + B * (g + Tc / m)
    res17b = -params["alpha"] * Gamma
    res18a = Gamma - np.linalg.norm(Tc)
    res18b = params["rho2"] - Gamma
    res18c = Gamma - params["rho1"]
    res19 = np.dot(n.T, Tc) - math.cos(theta) * Gamma
    res20 = error_margin - np.linalg.norm(E * r_func(tf) - q)

    c = np.array([res5a, res5b, res20]) #test
    return c


# def problem3(des_vars, params):
#     """ Problem 3: convex relaxed min landing error problem """
#     tf = des_vars[0]
#     Tc = des_vars[1]
#     Gamma = des_vars[2]
#
#     f3 = np.linalg.norm(E * r_func(tf) - q)
#
#     # add gradient
#     g3 = np.zeros(2)
#
#     return f3, g3

# def problem4(Tc, params):
#     """problem 4: convex relaxed min fuel problem"""
#     tf = params["tf"]
#
#     Gamma = np.linalg.norm(Tc)
#     f4 = np.sum(Gamma[0:tf], axis=0)
#
#     return f4

def problem4_discrete(Tc_array, state0, params):
    """problem 4: convex relaxed min fuel problem; discrete version"""
    # Calculate the total objective by summing the costs at each time step
    f4 = 0
    tf = int(state0[-1])
    for Tc_i in Tc_array:
        # Compute the cost at a specific time step using Tc_i
        Gamma = np.linalg.norm(Tc_i)
        cost_at_step_i = np.sum(Gamma[0:tf], axis=0)
        f4 += cost_at_step_i
    return f4

def constraints_func_discrete(Tc_array, state0, params):
    """ Discrete constraints at each time step """
    all_constraints = []
    omega = np.array([2.53e-5, 0, 6.62e-5])
    A = A_func(omega)  # 6x6
    g = np.array([[-3.71], [0], [0]])  # constant gravity vector
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3)
    E = np.zeros((2, 3))
    E[0, :] = np.array([0, 1, 0])
    E[1, :] = np.array([0, 0, 1])

    for i in range(len(Tc_array)):
        # Extract variables at the specific time step
        Tc_i = Tc_array[i]
        x_i = state0[i]
        r_i = x_i[0:3]
        rdot_i = x_i[3:6]
        # Extract necessary parameters from 'params' dictionary
        # Vmax = params["Vmax"]
        # error_margin = params["error_margin"]
        # mf = params["mf"]
        # m0 = params["m0"]
        # omega = params["omega"]
        # rho1 = params["rho1"]
        # rho2 = params["rho2"]
        # alpha = params["alpha"]
        # g = params["gravity"]
        alpha = 5e-4,  # s/m
        q = np.zeros((3,)) #landing target coordinates
        omega = np.array([2.53e-5, 0, 6.62e-5])
        gamma = math.pi / 4  # slide slope angle (0, pi/2)
        m0 = 2000  # kg
        mf = 300  # kg
        theta = math.pi/4 #thrust pointing constraint angle
        Vmax = 100 #m/s physically reasonable max


        """ Residuals of the constraints at this time step """
        res5a = Vmax - np.linalg.norm(x_i[3:6])  # x_i[3:6] contains v_x, v_y, v_z
        # You'll need to specify r_func(tf) and the values of x, y, z at this time step
        res5b = np.linalg.norm(E @ r_i - c.T @ (np.array([x_i[0], x_i[1], x_i[2]]) - r_i))
        res7b = x_i[6] - m0 - mf
        res9a = np.dot(np.array([1, 0, 0]), np.array([x_i[0], x_i[1], x_i[2]]))
        res9b = np.array([x_i[3], x_i[4], x_i[5]])  # constraint: rdot(tf) = 0
        res17a = A_func(omega) @ x_i + B @ (g + Tc_i / x_i[6])
        res17b = -alpha * np.linalg.norm(Tc_i)
        res18a = np.linalg.norm(Tc_i) - rho1
        res18b = rho2 - np.linalg.norm(Tc_i)
        res18c = np.linalg.norm(Tc_i) - rho1
        res19 = np.dot(np.array([1, 0, 0]), Tc_i) - np.cos(params["theta"]) * np.linalg.norm(Tc_i)
        res20 = error_margin - np.linalg.norm(E @ r_i - params["q"])

        # Append the constraints at this time step to the list
        constraints_at_step_i = [res5a, res5b, res7b, res9a] + res9b.tolist() + res17a.tolist() + [res17b, res18a, res18b, res18c, res19, res20]
        all_constraints.extend(constraints_at_step_i)

    return np.array(all_constraints)

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

    """Solve Problem 4 Optimization Problem"""
    #try optimizing ONLY FOR THE tf first!!!!
    Tc0 = 0.5*24000*np.ones((3,))
    state0 = np.append(x0, m0)
    constraints = {'type': 'ineq', 'fun': constraints_func_discrete, 'args':(state0, params,)}
    f_star = scipy.optimize.minimize(problem4_discrete, Tc0, args=(state0, params,), constraints=[constraints])
    #results = scipy.optimize.minimize(problem4, Tc0, method='BFGS', options={'disp': True})

    """ Propogate Dynamics"""
    z0 = np.append(x0, m0)
    tarray = np.linspace(0, 10, 1000)
    Tc = 0.5 * Tmax  # assuming constant thrust over this short timer
    z = scipy.integrate.odeint(rdot_func, z0, tarray, args=(Tc, params))
    x = z[:, 0:6]
    m = z[:, 6]
    plot_trajectory(tarray, x)


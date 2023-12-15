import numpy as np
import openmdao.api as om
import dymos as dm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import build_pyoptsparse
import pyoptsparse
import pickle

if __name__ == '__main__':

    """ Import for formulation1 """
    # sol = om.CaseReader('dymos_solution.db').get_case('final')
    # sim = om.CaseReader('dymos_simulation.db').get_case('final')
    # with open('prob3_dataset.pkl', 'rb') as db_file:
    #     [sol3, sim3] = pickle.load(db_file)
    # with open('prob4_dataset.pkl', 'rb') as db_file:
    #     [sol4, _] = pickle.load(db_file)
    # sol = sol4

    """ extract sim outputs into numpy arrays """
    # time = sol.get_val("traj.phase0.timeseries.time")
    # iters = np.floor(np.linspace(0, len(time), len(time)))
    # x = sol.get_val("traj.phase0.timeseries.x")
    # y = sol.get_val("traj.phase0.timeseries.y")
    # z = sol.get_val("traj.phase0.timeseries.z")
    # v_x = sol.get_val("traj.phase0.timeseries.v_x")
    # v_y = sol.get_val("traj.phase0.timeseries.v_y")
    # v_z = sol.get_val("traj.phase0.timeseries.v_z")
    # T_x = sol.get_val("traj.phase0.timeseries.T_x")
    # T_y = sol.get_val("traj.phase0.timeseries.T_y")
    # T_z = sol.get_val("traj.phase0.timeseries.T_z")
    # Gamma = sol.get_val("traj.phase0.timeseries.Gamma")
    # mass = sol.get_val("traj.phase0.timeseries.m")
    # obj3 = sol.get_val("traj.phase0.rhs_all.obj3")
    #obj4 = sol.get_val("traj.phase0.rhs_all.obj4")

    """ import directly """
    # with open('prob3_nparrays.pkl', 'rb') as db_file:
    #     [time, iters, x, y, z, v_x, v_y, v_z,
    #                  T_x, T_y, T_z, Gamma, mass, obj3] = pickle.load(db_file)
    # with open('prob4_nparrays.pkl', 'rb') as db_file:
    #     [time, iters, x, y, z, v_x, v_y, v_z,
    #                  T_x, T_y, T_z, Gamma, mass, obj4] = pickle.load(db_file)
    with open('form2_nparrays.pkl', 'rb') as db_file:
        [time, iters, x, y, z, v_x, v_y, v_z,
         T_x, T_y, T_z, Gamma, mass, obj4] = pickle.load(db_file)

    print("tf* = ", time[-1])
    print("Gamma* = ", Gamma[-1])
    print("Tx* = ", ", Ty* = ", ", Tz* = ")
    print("landing position = ", y[-1], ', ', z[-1], ' m')
    #print("landing error (obj3) = ", obj3[-1], ' m')
    print("Total Impulse (obj4) = ", obj4[-1], 'Newton s')

    """ PLOTS """

    # plot surface trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='k', label="Trajectory")
    start_point = [x[0], y[0], z[0]]
    end_point = [x[-1], y[-1], z[-1]]
    print("dp3* = ", end_point)
    ax.scatter(*start_point, color='red', label='Start Position')
    ax.scatter(*end_point, color='green', label='End Position')
    ax.set_xlabel('X-position (m)')
    ax.set_ylabel('Y-position (m)')
    ax.set_zlabel('Z-position (m)')
    ax.grid(True)
    ax.set_title('Lander Trajectory')
    plt.legend()
    plt.show()
    plt.close()
    check = 1

    # plot Gamma
    plt.figure()
    plt.plot(time, mass, marker='o', linestyle='dashed')
    plt.xlabel("time (s)")
    plt.ylabel('Gamma (N)')
    #plt.suptitle("Thrust Upper Bound: Gamma")
    plt.title("Thrust Upper Bound: Gamma")
    plt.grid(color='k', linewidth=0.5)
    plt.show()
    plt.close()

    # # plot convergence obj3
    # plt.figure()
    # plt.plot(iters, obj3, marker='o', linestyle='dashed')
    # plt.yscale('log')
    # plt.xlabel("Iteration number")
    # plt.ylabel('obj3 (m)')
    # #plt.suptitle("Convergence of Landing Error")
    # plt.title("Convergence of Landing Error")
    # plt.grid(color='k', linewidth=0.5)
    # plt.show()
    # plt.close()

    # plot convergence obj4
    plt.figure()
    plt.plot(time, obj4, marker='o', linestyle='dashed')
    plt.yscale('log')
    plt.xlabel("Iteration number")
    plt.ylabel('obj4 (kg)')
    # plt.suptitle("Convergence of Landing Error")
    plt.title("Convergence of Fuel Consumption")
    plt.grid(color='k', linewidth=0.5)
    plt.show()
    plt.close()
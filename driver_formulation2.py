import numpy as np
import openmdao.api as om
import dymos as dm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from classes_formulation2 import LanderODE2
import matplotlib.pyplot as plt

# import build_pyoptsparse
# import pyoptsparse

if __name__ == '__main__':
    """ PROBLEM FORMULATION 2: use problem 3 as a constraint for problem 4"""
    """ BUILD PROBLEM """

    # initialization
    m0 = 2000  # kg
    mf = 300  # kg
    t0 = 0
    tmax = 75
    tf = 10  # THIS IS A DESIGN VARIABLE
    r0 = np.array([2400, 450, -330])  # m
    rdot0 = np.array([-10, -40, 10])  # m/s
    x0 = np.zeros((6,))
    x0[0:3] = r0
    x0[3:6] = rdot0
    q = 0  # m, target landing site

    # Instantiate the problem, add the driver, and allow it to use coloring
    p = om.Problem(model=om.Group())
    # p.driver = om.pyOptSparseDriver() #cursed
    p.driver = ScipyOptimizeDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = 'SLSQP'

    # Instantiate the trajectory and add a phase to it
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase0 = traj.add_phase('phase0',
                            dm.Phase(ode_class=LanderODE2,
                                     transcription=dm.Radau(num_segments=15, order=3)))
    phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)  # maximum duration of simulation?

    """ ADD STATES """
    phase0.add_state('x', fix_initial=True, fix_final=True, units='m',
                     rate_source='xdot', lower=0, ref0=2000, ref=10)
    phase0.add_state('y', fix_initial=True, fix_final=True, units='m',
                     rate_source='ydot', lower=0, ref0=2000, ref=10)
    phase0.add_state('z', fix_initial=True, fix_final=True, units='m',
                     rate_source='zdot', lower=0, ref0=2000, ref=10)
    phase0.add_state('v_x', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_xdot', lower=0, ref0=2000, ref=10)
    phase0.add_state('v_y', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_ydot', lower=0, ref0=2000, ref=10)
    phase0.add_state('v_z', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_zdot', lower=0, ref0=2000, ref=10)
    phase0.add_state('m', fix_initial=True, fix_final=True, units='kg',
                     rate_source='mdot', lower=0, ref0=2000, ref=10)

    ## add states related to design variables
    # phase0.add_state('obj4', fix_initial=False, fix_final=False, rate_source='obj4dot') #do we need a obj4dot?
    # phase0.add_state('Gamma', fix_initial=False, fix_final=False)
    phase0.add_control('Gamma', units='N', opt=True)  # can add upper and lower limits if desired
    # phase0.add_control('Tc', units='N', opt=True, lower=min_Tc, upper=max_Tc)
    phase0.add_parameter('tf', units='s', opt=True)
    phase0.add_control('T_x', units='N', opt=True, lower=4800, upper=19200, )  # control input
    phase0.add_control('T_y', units='N', opt=True, lower=4800, upper=19200, )  # control input
    phase0.add_control('T_z', units='N', opt=True, lower=4800, upper=19200, )  # control input

    ## add states related to constraints
    # if the state is not fixed at start/end --> need to define rate_source
    # rate_source wrt time, string to path of ODE output which gives rate of state var
    phase0.add_state('res5a', fix_initial=False, rate_source='res5a', units='m/s')  # constraint 5a
    phase0.add_state('res5b', fix_final=False, rate_source='res5b', units='m')  # constraint 5b, dimensionless
    phase0.add_state('res7b', fix_initial=False, rate_source='res7b', units='kg')  # constraint 7b
    phase0.add_state('res9a', fix_initial=False, rate_source='res9a', units='m')  # constraint 9a
    phase0.add_state('res9b', fix_final=False, rate_source='res9b', units='m/s')  # constraint 9b
    phase0.add_state('res17a', fix_initial=False, rate_source='res17a', units='m/s')  # constraint 17a
    phase0.add_state('res17b', fix_initial=False, rate_source='res17b', units='kg/s')  # constraint 17b
    phase0.add_state('res18a', fix_initial=False, rate_source='res18a', units='N')  # constraint 18a
    phase0.add_state('res18b', fix_initial=False, rate_source='res18b', units='N')  # constraint 18b
    phase0.add_state('res18c', fix_initial=False, rate_source='res18c', units='N')  # constraint 18c
    phase0.add_state('res19', fix_initial=False, rate_source='res19', units='N')  # constraint 19
    phase0.add_state('res20', fix_initial=False, rate_source='res20', units='m')  # constraint 20

    """ CONSTRAINTS """
    # Add Constraints
    # for constraining expressions, the derivatives are calculated with complex step not analytic expressions
    phase0.add_boundary_constraint("m", loc="initial", equals=m0)  # constraint 7a
    phase0.add_boundary_constraint("res7b", loc="initial")
    phase0.add_boundary_constraint("x", loc="initial", equals=r0[0])  # constraint 8a
    phase0.add_boundary_constraint("y", loc="initial", equals=r0[1])  # constraint 8a
    phase0.add_boundary_constraint("z", loc="initial", equals=r0[2])  # constraint 8a
    phase0.add_boundary_constraint("v_x", loc="initial", equals=rdot0[0])  # constraint 8b
    phase0.add_boundary_constraint("v_y", loc="initial", equals=rdot0[1])  # constraint 8b
    phase0.add_boundary_constraint("v_z", loc="initial", equals=rdot0[2])  # constraint 8b
    phase0.add_boundary_constraint("res9a", loc="final")
    phase0.add_boundary_constraint("res9b", loc="final")  # constraint 9b
    phase0.add_boundary_constraint('res17b=-alpha*Gamma', loc="initial",
                                   equals=5e-4 * 0.8 * 24000)  # update with mdot(0)
    phase0.add_boundary_constraint('res20', loc="final")  # constraint 20

    ## Add Path constraints
    # phase0.add_timeseries_output('q', shape=(1,))
    phase0.add_path_constraint("res5a")
    phase0.add_path_constraint("res5b")
    phase0.add_path_constraint('res17a')  # evaluated as a state w/ dynamics expression
    phase0.add_path_constraint('Gamma >= np.linalg.norm(Tc)')  # constraint/res18a
    phase0.add_path_constraint('rho2 >= Gamma')  # constraint/res18b
    phase0.add_path_constraint('Gamma >= rho1')  # constraint/res18c
    phase0.add_path_constraint('res19')  # constraint 19

    # Add constraints to the system itself, initial conditions
    # p.model.add_constraint("m", lower=m0) #constraint 7a
    # p.model.add_constraint("r", lower=r0) #constraint 8a
    # p.model.add_constraint("rdot", lower=rdot0) #constraint 8b
    # p.model.add_constraint("res7b", lower=m0, upper=mf)

    """ OBJECTIVE """
    phase0.add_objective('obj4', loc='final')
    # phase0.add_objective('Tc', loc='final', ref=-0.01)
    # phase0.add_objective('Gamma', loc='final', ref=-0.01)
    # phase0.add_objective('q', loc='final', ref=-0.01)

    """ ADD PARAMETERS """
    # ADD ERROR_MARGIN HERE
    # phase0.add_control('Tc', units='N', opt=True, lower=min_Tc, upper=max_Tc)
    # phase0.add_parameter('Gamma', units='N', opt=True, lower=min_Gamma, upper=max_Gamma)

    objectives = phase0.get_objectives()
    print(objectives)

    design_vars = phase0.get_design_vars()
    print(design_vars)

    constraints = phase0.get_constraints()
    print(constraints)

    p.setup(check=True)

    """ INITIAL CONDITIONS OF THE STATES: Must be done after setup 'check' == True """
    p.set_val('traj.phase0.t_initial', t0, units='s')
    p.set_val('traj.phase0.t_duration', tmax, units='s')  # redundant with instantiating trajectory?
    # p.set_val('traj.phase0.states:r', np.array([2400, 450, -330]), units='m') #POSSIBLE TO SET AN IC AS AN ARRAY
    # p.set_val('traj.phase0.states:rdot', np.array([-10, -40, 10]), units='m/s')
    p.set_val('traj.phase0.states:x', 2400)
    p.set_val('traj.phase0.states:y', 450)
    p.set_val('traj.phase0.states:z', -330)
    p.set_val('traj.phase0.states:v_x', -10)
    p.set_val('traj.phase0.states:v_y', -40)
    p.set_val('traj.phase0.states:v_z', 10)
    p.set_val('traj.phase0.states:m', 2000)
    p.set_val('traj.phase0.controls:T_x', 0.2 * 24000)
    p.set_val('traj.phase0.controls:T_y', 0.2 * 24000)
    p.set_val('traj.phase0.controls:T_z', 0.2 * 24000)

    objectives = phase0.get_objectives()
    print(objectives)

    design_vars = phase0.get_design_vars()
    print(design_vars)

    constraints = phase0.get_constraints()
    print(constraints)

    """ RUN THE DRIVER """
    dm.run_problem(p, simulate=True)

    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')

    """ PLOT """
    plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
                   'time (s)', 'alpha (rad)'),
                  ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.beta',
                   'time (s)', 'beta (rad)'),
                  ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.theta',
                   'time (s)', 'theta (rad)'),
                  ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q',
                   'time (s)', 'q (Btu/ft**2/s)')], title='Reentry Solution', p_sol=sol,
                 p_sim=sim)

    plt.show()

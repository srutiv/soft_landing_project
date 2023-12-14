import numpy as np
import openmdao.api as om
import dymos as dm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from classes_formulation import LanderODE_form2
import matplotlib.pyplot as plt
import build_pyoptsparse
import pyoptsparse
import pickle

if __name__ == '__main__':
    """ PROBLEM FORMULATION 2: use problem 3 as a constraint for problem 4"""

    """ BUILD PROBLEM """
    # initialization
    m0 = 2000  # kg
    mf = 300  # kg
    r0 = np.array([2400, 450, -330])  # m
    rdot0 = np.array([-10, -40, 10])  # m/s
    x0 = np.zeros((6,))
    x0[0:3] = r0
    x0[3:6] = rdot0
    q = np.array([0, 0])  # m, target landing site
    t0 = 0
    Tmax = 24000 #N
    tmax = 75
    Vmax = 100 #m/s

    # Instantiate the problem, add the driver, and allow it to use coloring
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    #p.driver = ScipyOptimizeDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = 'SLSQP'

    # Instantiate the trajectory and add a phase to it
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase0 = traj.add_phase('phase0',
                            dm.Phase(ode_class=LanderODE_form2,
                            transcription=dm.Radau(num_segments=20, order=3)))
    tx = phase0.options['transcription']
    phase0.set_time_options(fix_initial=True, units='s', duration_bounds=(10, 150)) #maximum duration of simulation?

    ivc_x = p.model.add_subsystem('ivc_xind', om.IndepVarComp(), promotes_outputs=['*'])
    ivc_x.add_output('x_tf_ind', shape=(tx.grid_data.subset_num_nodes['control_input']), units='m')
    p.model.add_design_var('x_tf_ind', units='m', lower=0 )

    ivc_y = p.model.add_subsystem('ivc_yind', om.IndepVarComp(), promotes_outputs=['*'])
    ivc_y.add_output('y_tf_ind', shape=(tx.grid_data.subset_num_nodes['control_input']), units='m')
    p.model.add_design_var('y_tf_ind', units='m')

    ivc_z = p.model.add_subsystem('ivc_zind', om.IndepVarComp(), promotes_outputs=['*'])
    ivc_z.add_output('z_tf_ind', shape=(tx.grid_data.subset_num_nodes['control_input']), units='m')
    p.model.add_design_var('z_tf_ind', units='m')

    """ ADD STATES """
    phase0.add_state('x', fix_initial=True, fix_final=True, units='m',
                     rate_source='xdot', lower=0)
    phase0.add_state('y', fix_initial=True, fix_final=False, units='m',
                     rate_source='ydot')
    phase0.add_state('z', fix_initial=True, fix_final=False, units='m',
                     rate_source='zdot')
    phase0.add_state('v_x', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_xdot')
    phase0.add_state('v_y', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_ydot')
    phase0.add_state('v_z', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='v_zdot')
    phase0.add_state('m', fix_initial=True, fix_final=False, units='kg',
                     rate_source='mdot', lower=m0-mf)
    phase0.add_state('obj4', fix_initial=True, fix_final=False) # rate source obtained in ode

    ## add states related to design variables
    phase0.add_control('Gamma', units='N', opt=True, lower=4800, upper=19200, )  # can add upper and lower limits if desired
    phase0.add_control('T_x', units='N', opt=True)  # control input
    phase0.add_control('T_y', units='N', opt=True)  # control input
    phase0.add_control('T_z', units='N', opt=True)  # control input

    """ CONSTRAINTS """
    # Add Boundary and Path Constraints
    # for constraining expressions, the derivatives are calculated with complex step not analytic expressions
    phase0.add_boundary_constraint("constraint5b", loc="final", equals=0)
    phase0.add_boundary_constraint('res7b = m - 1700', loc='final', lower=0)
    phase0.add_boundary_constraint("constraint20", loc="final", upper=0) #Modified so upper bound is 0, need to manually set error

    ## Add Path constraints
    phase0.add_path_constraint("constraint5a", lower=0)
    phase0.add_path_constraint("constraint5b", upper=-1e-6)
    phase0.add_path_constraint("constraint18a", lower=1e-6)
    phase0.add_path_constraint('constraint19', lower=1e-6)  # constraint 19
    #phase0.add_path_constraint("constraint20", upper=-1e-6)

    """ OBJECTIVE """
    phase0.add_objective('obj4', loc='final') #possibly add a ref or normalize

    p.setup(check=True)

    """ INITIAL CONDITIONS: setting up an IC for x at every time node for optimizer to start from. spans the whole timeline """
    p['traj.phase0.t_initial'] = t0 #adding a value to fix the initial time to
    p['traj.phase0.t_duration'] = 60.0
    p.set_val('traj.phase0.states:x', phase0.interp('x', [2400, 0]), units='m')
    p.set_val('traj.phase0.states:y', phase0.interp('y', [450, 0]), units='m')
    p.set_val('traj.phase0.states:z', phase0.interp('z', [-330, 0]), units='m')
    p.set_val('traj.phase0.states:v_x', phase0.interp('v_x', [-10, 0]), units='m/s')
    p.set_val('traj.phase0.states:v_y', phase0.interp('v_y', [-40, 0]), units='m/s')
    p.set_val('traj.phase0.states:v_z', phase0.interp('v_z', [10, 0]), units='m/s')
    p.set_val('traj.phase0.states:m', phase0.interp('m', [m0, m0-0.5*mf]), units='kg') #do we want to fix mf here?
    p.set_val('traj.phase0.controls:T_x', phase0.interp('T_x', [0.5*Tmax, 0.5*Tmax]), units='N')
    p.set_val('traj.phase0.controls:T_y', phase0.interp('T_y', [0.5*Tmax, 0.5*Tmax]), units='N')
    p.set_val('traj.phase0.controls:T_z', phase0.interp('T_z', [0.5*Tmax, 0.5*Tmax]), units='N')
    p.set_val('traj.phase0.controls:Gamma', phase0.interp('Gamma', [0.7 * Tmax, 0.7 * Tmax]), units='N') 

    print("check1")
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

    with open('form2_dataset.pkl', 'wb') as db_file:
        pickle.dump([sol, sim], file=db_file)

    time = sol.get_val("traj.phase0.timeseries.time")
    iters = np.floor(np.linspace(0, len(time), len(time)))
    x = sol.get_val("traj.phase0.timeseries.x")
    y = sol.get_val("traj.phase0.timeseries.y")
    z = sol.get_val("traj.phase0.timeseries.z")
    v_x = sol.get_val("traj.phase0.timeseries.v_x")
    v_y = sol.get_val("traj.phase0.timeseries.v_y")
    v_z = sol.get_val("traj.phase0.timeseries.v_z")
    T_x = sol.get_val("traj.phase0.timeseries.T_x")
    T_y = sol.get_val("traj.phase0.timeseries.T_y")
    T_z = sol.get_val("traj.phase0.timeseries.T_z")
    Gamma = sol.get_val("traj.phase0.timeseries.Gamma")
    mass = sol.get_val("traj.phase0.timeseries.m")
    obj4 = sol.get_val("traj.phase0.timeseries.obj4")

    with open('form2_nparrays.pkl', 'wb') as db_file:
        pickle.dump([time, iters, x, y, z, v_x, v_y, v_z,
                     T_x, T_y, T_z, Gamma, mass, obj4], file=db_file)
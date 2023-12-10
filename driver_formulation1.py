import numpy as np
import openmdao.api as om
import dymos as dm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver

from classes_formulation1 import LanderODE
import matplotlib.pyplot as plt
#import build_pyoptsparse
#import pyoptsparse

if __name__ == '__main__':

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
    #p.driver = om.pyOptSparseDriver() #cursed
    p.driver = ScipyOptimizeDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = 'SLSQP'

    # Instantiate the trajectory and add a phase to it
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase0 = traj.add_phase('phase0',
                            dm.Phase(ode_class=LanderODE,
                            transcription=dm.Radau(num_segments=20, order=3)))
    tx = phase0.options['transcription']
    phase0.set_time_options(fix_initial=True, units='s', duration_bounds=(10, 150)) #maximum duration of simulation?

    ## Constraint 5b
    # Add an indep var comp to provide the external control values
    # Add the output to provide the values of theta at the control input nodes of the transcription
    # Add this external control as a design variable
    # Connect this to controls:tf in the appropriate phase.
    # connect calls are cached, so we can do this before we actually add the trajectory to the problem.

    ivc_x = p.model.add_subsystem('ivc_xind', om.IndepVarComp(), promotes_outputs=['*'])
    ivc_x.add_output('x_tf_ind', shape=(tx.grid_data.subset_num_nodes['control_input']), units='m')
    p.model.add_design_var('x_tf_ind', units='m' )
    # p.model.connect('x_tf_ind', 'traj.phase0.controls:x_tf_ind')

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

    ## add states related to design variables
    phase0.add_control('Gamma', units='N', opt=True, lower=4800, upper=19200, )  # can add upper and lower limits if desired
    # phase0.add_control('Tc', units='N', opt=True, lower=min_Tc, upper=max_Tc)
    # phase0.add_parameter('tf', units='s', opt=True)
    # Use opt=False to allow it to be connected to an external source.
    # Arguments lower and upper are no longer valid for an input control.
    # phase0.add_control('x_tf_ind', targets=['x_tf_ind'], opt=False)
    phase0.add_control('T_x', units='N', opt=True)  # control input
    phase0.add_control('T_y', units='N', opt=True)  # control input
    phase0.add_control('T_z', units='N', opt=True)  # control input

    ## add states related to constraints
    #phase0.add_state('res5a', units='m/s') #constraint 5a
    # phase0.add_state('res7b', fix_initial=False) #constraint 7b
    # phase0.add_state('res9a', fix_initial=False) #constraint 9a
    # phase0.add_state('res9b', fix_initial=True)  # constraint 9b
    # phase0.add_state('res17a', fix_initial=False) #constraint 17a
    #phase0.add_state('res19', fix_initial=False) #constraint 19

    """ CONSTRAINTS """
    # Add Boundary and Path Constraints
    # for constraining expressions, the derivatives are calculated with complex step not analytic expressions
    #phase0.add_path_constraint('res5a', lower=0, upper=70, ref=70)
    # phase0.add_timeseries_output('q', shape=(1,))
    #phase0.add_boundary_constraint("res5a")
    phase0.add_boundary_constraint("constraint5b", loc="final", equals=0)
    phase0.add_boundary_constraint('res7b = m - 1700', loc='final', lower=0)
    #phase0.add_boundary_constraint("res9a")
    #phase0.add_boundary_constraint('res17a') #evaluated as a state w/ dynamics expression
    #phase0.add_boundary_constraint('res17b=-alpha*Gamma', loc="initial", equals=5e-4*0.8*24000) #update with mdot(0)
    # phase0.add_boundary_constraint('Gamma >= np.linalg.norm(Tc)') #constraint/res18a
    # phase0.add_boundary_constraint('rho2 >= Gamma') #constraint/res18b
    # phase0.add_boundary_constraint('Gamma >= rho1') #constraint/res18c
    #phase0.add_boundary_constraint('res19') #constraint 19

    ## Add Path constraints
    # phase0.add_timeseries_output('q', shape=(1,))
    phase0.add_path_constraint("constraint5a", lower=0)
    phase0.add_path_constraint("constraint5b", upper=-1e-6)
    phase0.add_path_constraint("constraint18a", lower=1e-6)
    #phase0.add_path_constraint('res17a')  # evaluated as a state w/ dynamics expression
    #phase0.add_path_constraint('Gamma >= np.linalg.norm(Tc)')  # res18a
    # phase0.add_path_constraint('rho2 >= Gamma')  # constraint/res18b
    # phase0.add_path_constraint('Gamma >= rho1')  # constraint/res18c
    phase0.add_path_constraint('constraint19', lower=1e-6)  # constraint 19

    # Add constraints to the system itself, initial conditions
    #p.model.add_constraint("m", lower=m0) #constraint 7a
    #p.model.add_constraint("r", lower=r0) #constraint 8a
    #p.model.add_constraint("rdot", lower=rdot0) #constraint 8b
    #p.model.add_constraint("res7b", lower=m0, upper=mf)

    """ OBJECTIVE """
    phase0.add_objective('obj3', loc='final') #possibly add a ref or normalize
    # phase0.add_objective('Tc', loc='final', ref=-0.01)
    # phase0.add_objective('Gamma', loc='final', ref=-0.01)
    # phase0.add_objective('q', loc='final', ref=-0.01)

    p.setup(check=True)

    """ INITIAL CONDITIONS: setting up an IC for x at every time node for optimizer to start from. spans the whole timeline """
    p['traj.phase0.t_initial'] = t0 #adding a value to fix the initial time to
    p['traj.phase0.t_duration'] = 60.0
    # p.set_val('traj.phase0.t_duration', tmax, units='s')  # redundant with instantiating trajectory?
    # p.set_val('traj.phase0.states:r', np.array([2400, 450, -330]), units='m') #POSSIBLE TO SET AN IC AS AN ARRAY
    # p.set_val('traj.phase0.states:rdot', np.array([-10, -40, 10]), units='m/s')
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
    #p.set_val('traj.phase0.controls:x_tf_ind', phase.interp('x_tf_ind', [90, 90]), units='deg')

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

    # """ PLOT """
    # plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
    #                'time (s)', 'alpha (rad)'),
    #               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.beta',
    #                'time (s)', 'beta (rad)'),
    #               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.theta',
    #                'time (s)', 'theta (rad)'),
    #               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q',
    #                'time (s)', 'q (Btu/ft**2/s)')], title='Reentry Solution', p_sol=sol,
    #              p_sim=sim)

    # plt.show()
import numpy as np
import openmdao.api as om
import dymos as dm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver

from lander_classes import LanderODE
import matplotlib.pyplot as plt
#import build_pyoptsparse
#import pyoptsparse

if __name__ == '__main__':
    """build and run the problem """

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
                            transcription=dm.Radau(num_segments=15, order=3)))

    phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)
    phase0.add_state('r', fix_initial=True, fix_final=True, units='m',
                     rate_source='rdot', lower=0, ref0=2000, ref=10)
    phase0.add_state('m', fix_initial=True, fix_final=True, units='kg',
                     rate_source='mdot', lower=0, ref0=2000, ref=1000)
    phase0.add_state("obj3", fix_initial=False, fix_final=False, rate_source='obj3dot')
    # phase0.add_state('rdot', fix_initial=True, fix_final=True, units='m/s',
    #                  rate_source='rddot', lower=0, ref0=2500, ref=25000)
    phase0.add_control('Tc', units='N', opt=True, lower=4800, upper=19200, )
    #phase0.add_control('beta', units='rad', opt=True, lower=-89 * np.pi / 180, upper=1 * np.pi / 180, )

    # Add Constraints
    # for constraining expressions, the derivatives are calculated with complex step not analytic expressions
    phase0.add_boundary_constraint('res17a') #evaluated as a state w/ dynamics expression
    phase0.add_boundary_constraint('res17b=-alpha*Gamma', loc="initial", equals=5e-4*0.8*24000) #update with mdot(0)
    phase0.add_boundary_constraint('Gamma >= np.linalg.norm(Tc)') #res18a
    phase0.add_boundary_constraint('rho2 >= Gamma') #res18b
    phase0.add_boundary_constraint('Gamma >= rho1') #res18c
    phase0.add_boundary_constraint('res19')
    # phase0.add_path_constraint('q', lower=0, upper=70, ref=70)
    # phase0.add_timeseries_output('q', shape=(1,))

    # Add Objective
    phase0.add_objective('obj3', loc='final')
    # phase0.add_objective('Tc', loc='final', ref=-0.01)
    # phase0.add_objective('Gamma', loc='final', ref=-0.01)
    # phase0.add_objective('q', loc='final', ref=-0.01)

    p.setup(check=True)
    objectives = case.get_objectives()
    print(objectives)

    design_vars = case.get_design_vars()
    print(design_vars)

    constraints = case.get_constraints()
    print(constraints)

    p.set_val('traj.phase0.t_initial', 0, units='s')
    p.set_val('traj.phase0.t_duration', 75, units='s')

    p.set_val('traj.phase0.states:r',
              phase0.interp('r', [2400, 450, -330]), units='m')
    p.set_val('traj.phase0.states:rdot',
              phase0.interp('rdot', [-10, -40, 10]), units='m/s')
    p.set_val('traj.phase0.states:m',
              phase0.interp('m', 2000), units='kg')
    p.set_val('traj.phase0.controls:Tc', #RANGE OF CONTROLS INPUTS?
              phase0.interp('Tc', ys=[0.2*24000, 0.8*24000]), units='N')

    # Run the driver
    dm.run_problem(p, simulate=True)

    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')

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
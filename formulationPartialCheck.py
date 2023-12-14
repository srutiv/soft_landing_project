import numpy as np
import openmdao.api as om
from classes_formulation1 import LanderODE

num_nodes = 3

p = om.Problem(model=om.Group())

ivc = p.model.add_subsystem('vars', om.IndepVarComp())

ivc.add_output('x', shape=(num_nodes,), units='m')
ivc.add_output('y', shape=(num_nodes,), units='m')
ivc.add_output('z', shape=(num_nodes,), units='m')
ivc.add_output('v_x', shape=(num_nodes,), units='m/s')
ivc.add_output('v_y', shape=(num_nodes,), units='m/s')
ivc.add_output('v_z', shape=(num_nodes,), units='m/s')
ivc.add_output('m', shape=(num_nodes,), units='kg')
ivc.add_output('T_x', shape=(num_nodes,), units='N')
ivc.add_output('T_y', shape=(num_nodes,), units='N')
ivc.add_output('T_z', shape=(num_nodes,), units='N')
ivc.add_output('Gamma', shape=(num_nodes,), units='N')
ivc.add_output('constraint5a', shape=(num_nodes,), units='N')
ivc.add_output('x_tf_ind', shape=(num_nodes,), units='m')
ivc.add_output('y_tf_ind', shape=(num_nodes,), units='m')
ivc.add_output('z_tf_ind', shape=(num_nodes,), units='m')

p.model.add_subsystem('ode', LanderODE(num_nodes=num_nodes))

p.model.connect('vars.x', 'ode.x')
p.model.connect('vars.y', 'ode.y')
p.model.connect('vars.z', 'ode.z')
p.model.connect('vars.v_x', 'ode.v_x')
p.model.connect('vars.v_y', 'ode.v_y')
p.model.connect('vars.v_z', 'ode.v_z')
p.model.connect('vars.m', 'ode.m')
p.model.connect('vars.T_x', 'ode.T_x')
p.model.connect('vars.T_y', 'ode.T_y')
p.model.connect('vars.T_z', 'ode.T_z')
p.model.connect('vars.Gamma', 'ode.Gamma')
p.model.connect('vars.x_tf_ind', 'ode.x_tf_ind')
p.model.connect('vars.y_tf_ind', 'ode.y_tf_ind')
p.model.connect('vars.z_tf_ind', 'ode.z_tf_ind')

p.setup(force_alloc_complex=True)


p.set_val('vars.x', np.random.uniform(1, 2400, num_nodes))
p.set_val('vars.y', np.random.uniform(1, 450, num_nodes))
p.set_val('vars.z', np.random.uniform(-330, 0, num_nodes))
p.set_val('vars.v_x', np.random.uniform(-10, 0, num_nodes))
p.set_val('vars.v_y', np.random.uniform(0, 30, num_nodes))
p.set_val('vars.v_z', np.random.uniform(0, 30, num_nodes))
p.set_val('vars.m', np.random.uniform(1800, 1900, num_nodes))
p.set_val('vars.T_x', np.random.uniform(.2*24000, .4*24000, num_nodes))
p.set_val('vars.T_y', np.random.uniform(.2*24000, .4*24000, num_nodes))
p.set_val('vars.T_z', np.random.uniform(.2*24000, .4*24000, num_nodes))
p.set_val('vars.Gamma', np.random.uniform(.6*24000, .8*24000, num_nodes))
p.set_val('vars.x_tf_ind', np.random.uniform(1e-3, 10, num_nodes))
p.set_val('vars.y_tf_ind', np.random.uniform(-.5, .5, num_nodes))
p.set_val('vars.z_tf_ind', np.random.uniform(-.5, .5, num_nodes))


p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)


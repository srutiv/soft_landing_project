import openmdao.api as om
import math
import numpy as np
from openmdao.api import Group
from openmdao.api import ExplicitComponent, Problem
from convex_functions import A_func, S_func


class LanderODE_obj4(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('eom', subsys=FlightDynamics1(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'm', 'T_x',
                                            'T_y', 'T_z', 'Gamma', ],
                           promotes_outputs=['xdot', 'ydot', 'zdot', 'v_xdot', 'v_ydot', 'v_zdot', 'mdot'])
        self.add_subsystem('obj4Dt', subsys=objective4(num_nodes=nn),
                           promotes_inputs=['Gamma', ],
                           promotes_outputs=['obj4Dot', ])
        self.add_subsystem('comp5a', subsys=constraint5a(num_nodes=nn),
                           promotes_inputs=['v_x', 'v_y', 'v_z', ],
                           promotes_outputs=['constraint5a', ])
        self.add_subsystem('comp5b', subsys=constraint5b(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'x_tf_ind', 'y_tf_ind', 'z_tf_ind'],
                           promotes_outputs=['constraint5b', ])
        self.add_subsystem('comp18a', subsys=constraint18a(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint18a', ])
        self.add_subsystem('comp19', subsys=constraint19(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint19', ])
        self.add_subsystem('comp20', subsys=constraint20(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z'],
                           promotes_outputs=['constraint20', ])


class LanderODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('eom', subsys=FlightDynamics1(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'm', 'T_x',
                                            'T_y', 'T_z', 'Gamma', ],
                           promotes_outputs=['xdot', 'ydot', 'zdot', 'v_xdot', 'v_ydot', 'v_zdot', 'mdot'])
        self.add_subsystem('obj3', subsys=objective3(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', ],
                           promotes_outputs=['obj3', ])
        self.add_subsystem('comp5a', subsys=constraint5a(num_nodes=nn),
                           promotes_inputs=['v_x', 'v_y', 'v_z', ],
                           promotes_outputs=['constraint5a', ])
        self.add_subsystem('comp5b', subsys=constraint5b(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'x_tf_ind', 'y_tf_ind', 'z_tf_ind'],
                           promotes_outputs=['constraint5b', ])
        self.add_subsystem('comp18a', subsys=constraint18a(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint18a', ])
        self.add_subsystem('comp19', subsys=constraint19(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint19', ])


class LanderODE_form2(Group):
    #evaluate obj4 with obj3 as a constraint

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('eom', subsys=FlightDynamics1(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'm', 'T_x',
                                            'T_y', 'T_z', 'Gamma', ],
                           promotes_outputs=['xdot', 'ydot', 'zdot', 'v_xdot', 'v_ydot', 'v_zdot', 'mdot'])
        self.add_subsystem('obj4Dt', subsys=objective4(num_nodes=nn),
                           promotes_inputs=['Gamma', ],
                           promotes_outputs=['obj4Dot', ])
        self.add_subsystem('comp5a', subsys=constraint5a(num_nodes=nn),
                           promotes_inputs=['v_x', 'v_y', 'v_z', ],
                           promotes_outputs=['constraint5a', ])
        self.add_subsystem('comp5b', subsys=constraint5b(num_nodes=nn),
                           promotes_inputs=['x', 'y', 'z', 'x_tf_ind', 'y_tf_ind', 'z_tf_ind'],
                           promotes_outputs=['constraint5b', ])
        self.add_subsystem('comp18a', subsys=constraint18a(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint18a', ])
        self.add_subsystem('comp19', subsys=constraint19(num_nodes=nn),
                           promotes_inputs=['T_x', 'T_y', 'T_z', 'Gamma'],
                           promotes_outputs=['constraint19', ])
        self.add_subsystem('comp20', subsys=constraint20_form2mod(num_nodes=nn), #modified using obj3 as a constraint
                           promotes_inputs=['x', 'y', 'z'],
                           promotes_outputs=['constraint20', ])

class constraint5a(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('v_x', val=np.ones(nn), desc='x velocity', units='m/s')
        self.add_input('v_y', val=np.ones(nn), desc='y velocity', units='m/s')
        self.add_input('v_z', val=np.ones(nn), desc='z velocity', units='m/s')

        # Derivatives of the equations of motions
        self.add_output('constraint5a', val=np.ones(nn), desc='Vmax constraint', units='m/s')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint5a', 'v_x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5a', 'v_y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5a', 'v_z', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        v_x = inputs['v_x']
        v_y = inputs['v_y']
        v_z = inputs['v_z']

        # constants
        Vmax = 100  # m/s

        constraint5a = Vmax - np.linalg.norm([v_x, v_y, v_z], axis=0)

        # constraint outputs
        outputs["constraint5a"] = constraint5a

    def compute_partials(self, inputs, J):
        # Unpack inputs 
        v_x = inputs['v_x']
        v_y = inputs['v_y']
        v_z = inputs['v_z']

        v_norm = np.linalg.norm([v_x, v_y, v_z], axis=0)

        # Divide by zero regulation epsilon
        epsilon = 1e-10

        # Assign Partials
        # Element wise ???
        J['constraint5a', 'v_x'] = v_x / (v_norm + epsilon)
        J['constraint5a', 'v_y'] = v_y / (v_norm + epsilon)
        J['constraint5a', 'v_z'] = v_z / (v_norm + epsilon)


class constraint5b(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('x', val=np.ones(nn), desc='x position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z position', units='m')
        self.add_input('x_tf_ind', val=np.ones(nn), desc='x_tf position', units='m')
        self.add_input('y_tf_ind', val=np.ones(nn), desc='y_tf position', units='m')
        self.add_input('z_tf_ind', val=np.ones(nn), desc='z_tf position', units='m')

        # Derivatives of the equations of motions
        self.add_output('constraint5b', val=np.ones(nn), desc='cone constraint', units='m/s')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint5b', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5b', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5b', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5b', 'x_tf_ind', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5b', 'y_tf_ind', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint5b', 'z_tf_ind', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        x_tf_ind = inputs['x_tf_ind']
        y_tf_ind = inputs['y_tf_ind']
        z_tf_ind = inputs['z_tf_ind']

        r = np.array([x, y, z])
        r_tf = np.array([x_tf_ind, y_tf_ind, z_tf_ind])

        # constants
        gamma = np.pi / 4
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
        e1 = np.array([1, 0, 0])  # axial direction
        c = e1 / math.tan(gamma)  # glide slope direction

        t2Test = np.dot(c.T, (r - r_tf))
        constraint5b = np.linalg.norm(np.dot(E, r - r_tf), axis=0) - np.dot(c.T, (r - r_tf))

        # constraint outputs
        outputs["constraint5b"] = constraint5b

    def compute_partials(self, inputs, J):
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        x_tf_ind = inputs['x_tf_ind']
        y_tf_ind = inputs['y_tf_ind']
        z_tf_ind = inputs['z_tf_ind']

        r = np.array([x, y, z])
        r_tf = np.array([x_tf_ind, y_tf_ind, z_tf_ind])

        # constants
        gamma = np.pi / 4
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
        e1 = np.array([1, 0, 0])  # axial direction
        c = e1 / math.tan(gamma)  # glide slope direction

        E_norm = np.linalg.norm(np.dot(E, r - r_tf), axis=0)

        # Divide by zero regulation epsilon
        epsilon = 1e-10
        E_norm = E_norm + epsilon

        # Intermediate values
        Jx = -1 / math.tan(gamma)
        Jy = (y - y_tf_ind) / E_norm
        Jz = (z - z_tf_ind) / E_norm

        # Assign Partials
        J['constraint5b', 'x'] = -1 / math.tan(gamma)
        J['constraint5b', 'x_tf_ind'] = 1 / math.tan(gamma)

        J['constraint5b', 'y'] = (y - y_tf_ind) / E_norm
        J['constraint5b', 'y_tf_ind'] = -(y - y_tf_ind) / E_norm

        J['constraint5b', 'z'] = (z - z_tf_ind) / E_norm
        J['constraint5b', 'z_tf_ind'] = -(z - z_tf_ind) / E_norm


class constraint18a(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('T_x', val=np.ones(nn), desc='Thrust x', units='N')
        self.add_input('T_y', val=np.ones(nn), desc='Thrust y', units='N')
        self.add_input('T_z', val=np.ones(nn), desc='Thrust z', units='N')
        self.add_input('Gamma', val=np.ones(nn), desc='Thrust norm', units='N')

        # Derivatives of the equations of motions
        self.add_output('constraint18a', val=np.ones(nn), desc='Thrust constraint', units='N')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint18a', 'T_x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint18a', 'T_y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint18a', 'T_z', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint18a', 'Gamma', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        Gamma = inputs['Gamma']

        constraint18a = Gamma - np.linalg.norm([T_x, T_y, T_z], axis=0)

        # constraint outputs
        outputs["constraint18a"] = constraint18a

    def compute_partials(self, inputs, J):
        # Unpack inputs
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        Gamma = inputs['Gamma']

        T_norm = np.linalg.norm([T_x, T_y, T_z], axis=0)

        # Divide by zero regulation epsilon
        epsilon = 1e-10
        T_norm = T_norm + epsilon

        # Assign Partials
        J['constraint18a', 'T_x'] = -T_x / T_norm
        J['constraint18a', 'T_y'] = -T_y / T_norm
        J['constraint18a', 'T_z'] = -T_z / T_norm
        J['constraint18a', 'Gamma'] = 1


class constraint19(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('T_x', val=np.ones(nn), desc='Thrust x', units='N')
        self.add_input('T_y', val=np.ones(nn), desc='Thrust y', units='N')
        self.add_input('T_z', val=np.ones(nn), desc='Thrust z', units='N')
        self.add_input('Gamma', val=np.ones(nn), desc='Thrust norm', units='N')

        # Derivatives of the equations of motions
        self.add_output('constraint19', val=np.ones(nn), desc='Thrust angle constraint', units='N')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint19', 'T_x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint19', 'T_y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint19', 'T_z', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint19', 'Gamma', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        Gamma = inputs['Gamma']

        # Constants
        nhat = np.array([1, 0, 0])  # normal direction
        theta = math.pi / 4  # thrust pointing constraint angle

        constraint19 = np.dot(nhat.T, np.array([T_x, T_y, T_z])) - math.cos(theta) * Gamma

        # constraint outputs
        outputs["constraint19"] = constraint19

    def compute_partials(self, inputs, J):
        # Unpack inputs
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        Gamma = inputs['Gamma']

        n = np.array([1, 0, 0])  # normal direction
        theta = math.pi / 4  # thrust pointing constraint angle

        # Assign Partials
        J['constraint19', 'T_x'] = n[0]
        J['constraint19', 'T_y'] = n[1]
        J['constraint19', 'T_z'] = n[2]
        J['constraint19', 'Gamma'] = -np.cos(theta)


class constraint20(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('x', val=np.ones(nn), desc='x tf position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y tf position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z tf position', units='m')

        # Derivatives of the equations of motions
        self.add_output('constraint20', val=np.ones(nn), desc='landing error', units='m')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint20', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint20', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint20', 'z', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        r = np.array([x, y, z])

        # constants
        dp3 = np.array([0.9999994, 1.0000002])
        q = np.array([0, 0])  # m, target landing site
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
       
        constraint20 = np.linalg.norm(np.dot(E, r) - q[:, np.newaxis], axis=0) - np.linalg.norm(dp3 - q)

        # constraint outputs
        outputs["constraint20"] = constraint20

    def compute_partials(self, inputs, J):
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        r_tf = np.array([x, y, z])

        # constants
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
        
        q = np.array([0, 0])  # m, target landing site
        norm_val = np.linalg.norm(np.array([y, z]) - q[:, np.newaxis], axis=0)

        # Divide by zero regulation epsilon
        epsilon = 1e-10
        norm_val = norm_val + epsilon

        # Assign Partials
        J['constraint20', 'x'] = 0
        J['constraint20', 'y'] = (y - q[0]) / norm_val
        J['constraint20', 'z'] = (z - q[1]) / norm_val

class constraint20_form2mod(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('x', val=np.ones(nn), desc='x_tf position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y_tf position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z_tf position', units='m')

        # Derivatives of the equations of motions
        self.add_output('constraint20', val=np.ones(nn), desc='landing error', units='m')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('constraint20', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint20', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('constraint20', 'z', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        r = np.array([x, y, z])

        # constants
        errtol = 1.5 # allowable error in position
        
        q = np.array([0, 0])  # m, target landing site
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])

        constraint20 = np.linalg.norm(np.dot(E, r) - q[:, np.newaxis], axis=0) - errtol #obj3 as a constraint
        #constraint20 = np.linalg.norm(np.dot(E, r_tf), axis=0) - np.linalg.norm(dp3 - q)

        # constraint outputs
        outputs["constraint20"] = constraint20

    def compute_partials(self, inputs, J):
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        r_tf = np.array([x, y, z])

        # constants
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
        q = np.array([0, 0])  # m, target landing site

        the_norm = np.linalg.norm(np.array([y, z]) - q[:, np.newaxis], axis=0)
        
        # Divide by zero regulation epsilon
        epsilon = 1e-10
        the_norm = the_norm + epsilon

        # Assign Partials
        J['constraint20', 'x'] = 0
        J['constraint20', 'y'] = (y - q[0]) / the_norm
        J['constraint20', 'z'] = (z - q[1]) / the_norm

class objective3(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', val=np.ones(nn), desc='x position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z position', units='m')

        # Derivatives of the equations of motions
        self.add_output('obj3', val=np.ones(nn), desc='minimum landing error', units='m')

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('obj3', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('obj3', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('obj3', 'z', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        # Constants
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])

        q = np.array([0, 0])  # m, target landing site
        r = np.array([x, y, z])

        obj3 = np.linalg.norm(np.dot(E, r) - q[:, np.newaxis], axis=0)

        # constraint outputs
        outputs["obj3"] = obj3

    def compute_partials(self, inputs, J):
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        q = np.array([0, 0])  # m, target landing site
        norm_val = np.linalg.norm(np.array([y, z]) - q[:, np.newaxis], axis=0)

        # Divide by zero regulation epsilon
        epsilon = 1e-10
        norm_val = norm_val + epsilon

        # Assign Partials
        J['obj3', 'x'] = 0
        J['obj3', 'y'] = (y - q[0]) / norm_val
        J['obj3', 'z'] = (z - q[1]) / norm_val


class objective4(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Gamma', val=np.ones(nn), desc='thrust bound', units='N')

        # Derivatives of the equations of motions
        self.add_output('obj4Dot', val=np.ones(nn), desc='Max Impulse', units='m', tags=['dymos.state_rate_source:obj4'])

        # Declare Partials of outputs wrt inputs
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('obj4Dot', 'Gamma', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        Gamma = inputs['Gamma']

        obj4Dot = Gamma  # integrate Gamma across time

        # constraint outputs
        outputs["obj4Dot"] = obj4Dot

    def compute_partials(self, inputs, J):
        # Unpack inputs
        Gamma = inputs['Gamma']

        # Assign Partials
        J['obj4Dot', 'Gamma'] = 1


class FlightDynamics1(ExplicitComponent):
    """
    Defines the flight dyanmics for the soft landing problem.

    References
    ----------
    .. [1] REPLACE WITH PROPER CITATION
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Equations of Motions
        self.add_input('x', val=np.ones(nn), desc='x position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z position', units='m')
        self.add_input('v_x', val=np.ones(nn), desc='x velocity', units='m/s')
        self.add_input('v_y', val=np.ones(nn), desc='y velocity', units='m/s')
        self.add_input('v_z', val=np.ones(nn), desc='z velocity', units='m/s')
        self.add_input('m', val=np.ones(nn), desc='mass', units='kg')
        self.add_input('T_x', val=np.ones(nn), desc='Thrust control x', units='N')
        self.add_input('T_y', val=np.ones(nn), desc='Thrust control y', units='N')
        self.add_input('T_z', val=np.ones(nn), desc='Thrust control z', units='N')
        self.add_input('Gamma', val=np.ones(nn), desc='Thrust control Bound', units='N')

        # Derivatives of the equations of motions
        self.add_output('xdot', val=np.ones(nn), desc='x position rate', units='m/s')
        self.add_output('ydot', val=np.ones(nn), desc='y position rate', units='m/s')
        self.add_output('zdot', val=np.ones(nn), desc='z position rate', units='m/s')
        self.add_output('v_xdot', val=np.ones(nn), desc='x velocity rate', units='m/s**2')
        self.add_output('v_ydot', val=np.ones(nn), desc='y velocity rate', units='m/s**2')
        self.add_output('v_zdot', val=np.ones(nn), desc='z velocity rate', units='m/s**2')
        self.add_output('mdot', val=np.ones(nn), desc='mass change rate', units='kg/s')

        # Add partial range
        partial_range = np.arange(nn, dtype=int)

        # Declare Partials of outputs wrt inputs
        self.declare_partials('xdot', 'v_x', rows=partial_range, cols=partial_range)

        self.declare_partials('ydot', 'v_y', rows=partial_range, cols=partial_range)

        self.declare_partials('zdot', 'v_z', rows=partial_range, cols=partial_range)

        self.declare_partials('v_xdot', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'v_y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'v_z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'T_x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'm', rows=partial_range, cols=partial_range)

        self.declare_partials('v_ydot', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'v_x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'v_z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'T_y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'm', rows=partial_range, cols=partial_range)

        self.declare_partials('v_zdot', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'v_x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'v_y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'T_z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'm', rows=partial_range, cols=partial_range)

        self.declare_partials('mdot', 'Gamma', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        """
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        v_x = inputs['v_x']
        v_y = inputs['v_y']
        v_z = inputs['v_z']
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        m = inputs['m']
        Gamma = inputs['Gamma']

        # Arrange into array for easy computing
        X = np.array([x, y, z, v_x, v_y, v_z])
        Tc = np.array([T_x, T_y, T_z])

        # Constants
        alpha = 5e-4  # s/m
        # g = np.array([[-3.71], [0], [0]])
        g = np.array([-3.71, 0, 0])
        omega = np.array([2.53e-5, 0, 6.62e-5])
        gamma = math.pi / 4  # slide slope angle (0, pi/2)
        A = A_func(omega)  # 6x6
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)
        E = np.zeros((2, 3))
        E[0, :] = np.array([0, 1, 0])
        E[1, :] = np.array([0, 0, 1])
        e1 = np.array([1, 0, 0])  # axial direction
        n = np.array([1, 0, 0])  # normal direction
        c = e1 / math.tan(gamma)  # glide slope direction

        # Compute Outputs using matrix math
        # XDOT = np.dot(A, X) + np.dot(B, (np.add(g, Tc / m)))
        XDOT = np.dot(A, X) + np.dot(B, g[:, np.newaxis] + (Tc / m))
        mdot = -1 * alpha * Gamma

        # Assign Outputs
        outputs["xdot"] = XDOT[0, :]
        outputs["ydot"] = XDOT[1, :]
        outputs["zdot"] = XDOT[2, :]
        outputs["v_xdot"] = XDOT[3, :]
        outputs["v_ydot"] = XDOT[4, :]
        outputs["v_zdot"] = XDOT[5, :]
        outputs["mdot"] = mdot

    def compute_partials(self, inputs, J):
        # Unpack inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        v_x = inputs['v_x']
        v_y = inputs['v_y']
        v_z = inputs['v_z']
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        m = inputs['m']
        Gamma = inputs['Gamma']

        # Arrange into array for easy computing
        X = np.array([[x], [y], [z], [v_x], [v_y], [v_z]])
        Tc = np.array([[T_x], [T_y], [T_z]])

        # Constants
        alpha = 5e-4  # s/m
        omega = np.array([2.53e-5, 0, 6.62e-5])
        A = A_func(omega)  # 6x6
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        # Reshape 'm' to (1, 1, 60) to make it compatible for broadcasting
        m_reshape = m.reshape(1, 1, -1)
        num_nodes = len(m)
        B_reshape = np.tile(B[:, :, np.newaxis], (1, 1, num_nodes))

        # Compute Partials using matrix math
        pXDOT_pX = A
        # pXDOT_pT = B/m
        pXDOT_pT = B_reshape / m_reshape  # 6x3xnum nodes
        # pXDOT_pm = -(1 / m ** 2) * np.dot(B_reshape, Tc) #6x1x6

        oneOverM = -(1 / m ** 2)
        TB = np.tensordot(B_reshape, Tc, axes=([1], [0]))
        pXDOT_pm = -(1 / m ** 2) * np.tensordot(B_reshape, Tc, axes=([1], [0]))  # tensor dot product
        # pXDOT_pm = np.swapaxes(pXDOT_pm, 1, 2)
        # pXDOT_pm = np.squeeze(pXDOT_pm) #squeeze unnecesasry dims
        pmdot_pGamma = -1 * alpha

        # Assign Partials
        # J['xdot', 'v_x'] = pXDOT_pX[0, 3]
        J['xdot', 'v_x'] = np.tile(pXDOT_pX[0, 3, np.newaxis], (1, 1, num_nodes))

        # J['ydot', 'v_y'] = pXDOT_pX[1, 4]
        J['ydot', 'v_y'] = np.tile(pXDOT_pX[1, 4, np.newaxis], (1, 1, num_nodes))

        # J['zdot', 'v_z'] = pXDOT_pX[2, 5]
        J['zdot', 'v_z'] = np.tile(pXDOT_pX[2, 5, np.newaxis], (1, 1, num_nodes))

        # J['v_xdot', 'x'] = pXDOT_pX[3, 0]
        J['v_xdot', 'x'] = np.tile(pXDOT_pX[3, 0, np.newaxis], (1, 1, num_nodes))
        # J['v_xdot', 'y'] = pXDOT_pX[3, 1]
        J['v_xdot', 'y'] = np.tile(pXDOT_pX[3, 1, np.newaxis], (1, 1, num_nodes))
        # J['v_xdot', 'z'] = pXDOT_pX[3, 2]
        J['v_xdot', 'z'] = np.tile(pXDOT_pX[3, 2, np.newaxis], (1, 1, num_nodes))
        # J['v_xdot', 'v_y'] = pXDOT_pX[3, 4]
        J['v_xdot', 'v_y'] = np.tile(pXDOT_pX[3, 4, np.newaxis], (1, 1, num_nodes))
        # J['v_xdot', 'v_z'] = pXDOT_pX[3, 5]
        J['v_xdot', 'v_z'] = np.tile(pXDOT_pX[3, 5, np.newaxis], (1, 1, num_nodes))
        J['v_xdot', 'T_x'] = pXDOT_pT[3, 0, :]  # 60,
        J['v_xdot', 'm'] = pXDOT_pm[3, 0, 0, :]

        # J['v_ydot', 'x'] = pXDOT_pX[4, 0]
        J['v_ydot', 'x'] = np.tile(pXDOT_pX[4, 0, np.newaxis], (1, 1, num_nodes))
        # J['v_ydot', 'y'] = pXDOT_pX[4, 1]
        J['v_ydot', 'y'] = np.tile(pXDOT_pX[4, 1, np.newaxis], (1, 1, num_nodes))
        # J['v_ydot', 'z'] = pXDOT_pX[4, 2]
        J['v_ydot', 'z'] = np.tile(pXDOT_pX[4, 2, np.newaxis], (1, 1, num_nodes))
        # J['v_ydot', 'v_x'] = pXDOT_pX[4, 3]
        J['v_ydot', 'v_x'] = np.tile(pXDOT_pX[4, 3, np.newaxis], (1, 1, num_nodes))
        # J['v_ydot', 'v_z'] = pXDOT_pX[4, 5]
        J['v_ydot', 'v_z'] = np.tile(pXDOT_pX[4, 5, np.newaxis], (1, 1, num_nodes))
        J['v_ydot', 'T_y'] = pXDOT_pT[4, 1, :]
        # J['v_ydot', 'm'] = pXDOT_pm[4, 0]
        J['v_ydot', 'm'] = pXDOT_pm[4, 0, 0, :]

        # J['v_zdot', 'x'] = pXDOT_pX[5, 0]
        J['v_zdot', 'x'] = np.tile(pXDOT_pX[5, 0, np.newaxis], (1, 1, num_nodes))
        # J['v_zdot', 'y'] = pXDOT_pX[5, 1]
        J['v_zdot', 'y'] = np.tile(pXDOT_pX[5, 1, np.newaxis], (1, 1, num_nodes))
        # J['v_zdot', 'z'] = pXDOT_pX[5, 2]
        J['v_zdot', 'z'] = np.tile(pXDOT_pX[5, 2, np.newaxis], (1, 1, num_nodes))
        # J['v_zdot', 'v_x'] = pXDOT_pX[5, 3]
        J['v_zdot', 'v_x'] = np.tile(pXDOT_pX[5, 3, np.newaxis], (1, 1, num_nodes))
        # J['v_zdot', 'v_y'] = pXDOT_pX[5, 4]
        J['v_zdot', 'v_y'] = np.tile(pXDOT_pX[5, 4, np.newaxis], (1, 1, num_nodes))
        J['v_zdot', 'T_z'] = pXDOT_pT[5, 2, :]
        # J['v_zdot', 'm'] = pXDOT_pm[5, 0]
        J['v_zdot', 'm'] = pXDOT_pm[5, 0, 0, :]

        J['mdot', 'Gamma'] = pmdot_pGamma

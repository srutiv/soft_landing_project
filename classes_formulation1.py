import openmdao.api as om
import math
import numpy as np
from openmdao.api import Group
from openmdao.api import ExplicitComponent, Problem
from convex_functions import A_func, S_func

############################DONT THINK THIS IS NEEDED ********************************************************"""
class LanderODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('eom', subsys=FlightDynamics(num_nodes=nn),
                           promotes_inputs=['r', 'rdot', 'm', 'Tc', 'omega', ],
                           promotes_outputs=['rddot', 'mdot', ])

################################USE THIS INSTEAD OR PUT THIS IN THE LANDERODE CLASS??????????"""
class FlightDynamics(ExplicitComponent):
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
        self.add_input('res7b', val=np.ones(nn), desc='constraint residual 7b', units='kg')
        self.add_input('res17a', val=np.ones(nn), desc='constraint residual 17a', units='m/s')
        self.add_input('res19', val=np.ones(nn), desc='constraint residual 19', units='N')

        # Derivatives of the equations of motions
        self.add_output('xdot', val=np.ones(nn), desc='x position rate', units='m/s')
        self.add_output('ydot', val=np.ones(nn), desc='y position rate', units='m/s')
        self.add_output('zdot', val=np.ones(nn), desc='z position rate', units='m/s')
        self.add_output('v_xdot', val=np.ones(nn), desc='x velocity rate', units='m/s**2')
        self.add_output('v_ydot', val=np.ones(nn), desc='y velocity rate', units='m/s**2')
        self.add_output('v_zdot', val=np.ones(nn), desc='z velocity rate', units='m/s**2')
        self.add_output('mDot', val=np.ones(nn), desc='mass change rate', units='kg/s')

        ####################### Time derivatives dont know if we need these? ####################################
        # self.add_output('Tc', val=np.ones(nn), desc='Tc optimal', units='N')
        # self.add_output('Gamma', val=np.ones(nn), desc='norm Tc optimal', units='N')
        # self.add_output('q', val=np.ones(nn), desc='final landing coordinates', units='m') 

        partial_range = np.arange(nn, dtype=int)

        # Declare Partials of outputs wrt inputs
        self.declare_partials('xdot', 'v_x', rows=partial_range, cols=partial_range)

        self.declare_partials('ydot', 'v_y', rows=partial_range, cols=partial_range)

        self.declare_partials('zdot', 'v_z', rows=partial_range, cols=partial_range)

        self.declare_partials('v_xdot', 'y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'v_y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'v_z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'T_x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_xdot', 'm', rows=partial_range, cols=partial_range)

        self.declare_partials('v_ydot', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'v_x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'v_z', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'T_y', rows=partial_range, cols=partial_range)
        self.declare_partials('v_ydot', 'm', rows=partial_range, cols=partial_range)

        self.declare_partials('v_zdot', 'x', rows=partial_range, cols=partial_range)
        self.declare_partials('v_zdot', 'y', rows=partial_range, cols=partial_range)
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
        Gamma = input['Gamma']
        # res17a = input["res17a"]
        # res19 = input["res19"]

        # Arrange into array for easy computing
        X = np.array([[x], [y], [z], [v_x], [v_y], [v_z]])
        Tc = np.array([[T_x], [T_y], [T_z]])

        # Constants
        alpha = 5e-4,  # s/m
        g = np.array([[-3.71], [0], [0]])
        omega = np.array([2.53e-5, 0, 6.62e-5])
        gamma = pi / 4  # slide slope angle (0, pi/2)
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
        XDOT = np.dot(A, X) + np.dot(B, (np.add(g, Tc / m)))
        mdot = -1 * alpha * Gamma

        # upate constraint residuals
        res5a = Vmax - np.linalg.norm(np.array[v_x, v_y, v_z])
        res5b = np.linalg.norm(E * r_func(tf) - c.T * (np.array[x, y, z] - r_func(tf)))  # HOW DO WE KNOW r_tf???
        res7b = m - m0 - mf
        res9a = np.dot(e1, np.array[x, y, z])
        res9b = np.array[v_x, v_y, v_z]  # constraint: rdot(tf) = 0
        res17a = A_func(omega) * x_func(tf) + B * (g + Tc / m)
        res19 = np.dot(n.T, Tc) - math.cos(theta) * Gamma
        res20 = error_margin - np.linalg.norm(E * r_func(tf) - q)

        # Assign Outputs
        outputs["xdot"] = XDOT[0, 0]
        outputs["xdot"] = XDOT[1, 0]
        outputs["xdot"] = XDOT[2, 0]
        outputs["v_xdot"] = XDOT[3, 0]
        outputs["v_xdot"] = XDOT[4, 0]
        outputs["v_xdot"] = XDOT[5, 0]
        outputs["mdot"] = mdot
        outputs["obj3"] = obj3

        # constraint outputs
        outputs["res5a"] = res5a
        outputs["res5b"] = res5b
        outputs["res7b"] = res7b
        outputs["res9a"] = res9a
        outputs["res17a"] = res17a
        outputs["res19"] = res19
        outputs["res20"] - res20

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
        alpha = 5e-4,  # s/m
        omega = np.array([2.53e-5, 0, 6.62e-5])
        A = A_func(omega)  # 6x6
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        # Compute Partials using matrix math
        pXDOT_pX = A
        pXDOT_pT = B / m
        pXDOT_pm = -(1 / m ** 2) * np.dot(B, Tc)
        pmdot_pGamma = -1 * alpha

        # Assign Partials
        J['xdot', 'v_x'] = pXDOT_pX[0, 3]

        J['ydot', 'v_y'] = pXDOT_pX[1, 4]

        J['zdot', 'v_z'] = pXDOT_pX[2, 5]

        J['v_xdot', 'y'] = pXDOT_pX[3, 1]
        J['v_xdot', 'z'] = pXDOT_pX[3, 2]
        J['v_xdot', 'v_y'] = pXDOT_pX[3, 4]
        J['v_xdot', 'v_z'] = pXDOT_pX[3, 5]
        J['v_xdot', 'T_x'] = pXDOT_pT[3, 0]
        J['v_xdot', 'm'] = pXDOT_pm[3, 0]

        J['v_ydot', 'x'] = pXDOT_pX[4, 0]
        J['v_ydot', 'z'] = pXDOT_pX[4, 2]
        J['v_ydot', 'v_x'] = pXDOT_pX[4, 3]
        J['v_ydot', 'v_z'] = pXDOT_pX[4, 5]
        J['v_ydot', 'T_y'] = pXDOT_pT[4, 0]
        J['v_ydot', 'm'] = pXDOT_pm[4, 0]

        J['v_zdot', 'x'] = pXDOT_pX[5, 0]
        J['v_zdot', 'y'] = pXDOT_pX[5, 1]
        J['v_zdot', 'v_x'] = pXDOT_pX[5, 3]
        J['v_zdot', 'v_y'] = pXDOT_pX[5, 4]
        J['v_zdot', 'T_z'] = pXDOT_pT[5, 0]
        J['v_zdot', 'm'] = pXDOT_pm[5, 0]

        J['mdot', 'Gamma'] = pmdot_pGamma

import openmdao.api as om
import numpy as np
from openmdao.api import Group
from openmdao.api import ExplicitComponent, Problem
from convex_functions import A_func, S_func

class LanderODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('eom', subsys=FlightDynamics(num_nodes=nn),
                           promotes_inputs=['r', 'rdot', 'm', 'Tc', 'omega', ],
                           promotes_outputs=['rddot', 'mdot', ])

class FlightDynamics(ExplicitComponent):
    """
    Defines the flight dynamics for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 247, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r', val=np.ones(nn), desc='position', units='m')
        self.add_input('rdot', val=np.ones(nn), desc='velocity', units='m/s')
        self.add_input('m', val=np.ones(nn), desc='mass', units='kg')
        self.add_input('Tc', val=np.ones(nn), desc='Thrust control', units='N')
        self.add_input('omega', val=np.ones(nn), desc='Thrust control', units='N')

        #self.add_output('r', val=np.ones(nn), desc='position', units='m')
        self.add_output('rddot', val=np.ones(nn), desc='', units='m/s')
        #self.add_output('tf_dot', val=np.ones(nn), desc='', units='s')
        #self.add_output('Tc_dot', val=np.ones(nn), desc='', units='N')
        self.add_output('mdot', val=np.ones(nn), desc='mdot', units='kg/s')
        self.add_output('obj3', val=np.ones(nn), desc='obj3')
        #self.add_output('Gamma_dot', val=np.ones(nn), desc='norm Tc optimal', units='N')

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials('rdot', 't', rows=partial_range, cols=partial_range)
        self.declare_partials('rddot', 't', rows=partial_range, cols=partial_range)

        self.declare_partials('mdot', 't', rows=partial_range, cols=partial_range)

        self.declare_partials('obj3dot', 't', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        dynamics function: lander modeled as a lumped parameter mass with Tc for control
        omega = np.ones((3,)) #vector of planets constant angular velocity
        """
        # unpack inputs
        x = inputs['x']
        rdot = inputs['rdot']
        m = inputs['m']
        Tc = inputs["Tc"]
        omega = inputs['omega']

        # constants
        Tmax = 24000 #N
        rho1 = 0.2 * Tmax  # thrust limit for throttle level 20%
        rho2 = 0.8 * Tmax
        alpha = 5e-4,  # s/m
        q_target = np.array([0, 0, 0]) #target_landing position

        x = np.zeros((6,))
        x[0:3] = r
        x[3:6] = rdot

        g = np.ones((3,))  # constant gravity vector
        S = S_func(omega)

        A = A_func(omega)  # 6x6
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        # projection matrix for landing position
        E = np.array([[0, 1, 0], [0, 0, 1]])

        obj3 = np.linalg.norm(np.dot(E, r) - q_target)

        xdot = np.dot(A, x) + np.dot(B, g + Tc / m)  # 6x1
        mdot = alpha * np.linalg.norm(Tc)
        zdot = np.append(xdot, mdot)

        outputs["rdot"] = xdot[0:3]
        outputs["rddot"] = xdot[3:6]
        outputs["mdot"] = mdot
        outputs["obj3"] = obj3

    # def compute_partials(self, inputs, J):
    #     v = inputs['v']
    #     gamma = inputs['gamma']
    #     theta = inputs['theta']
    #     lift = inputs['lift']
    #     h = inputs['h']
    #     beta = inputs['beta']
    #     psi = inputs['psi']
    #     g_0 = 32.174
    #     w = 203000
    #     R_e = 20902900
    #     mu = .14076539e17
    #     s_beta = np.sin(beta)
    #     c_beta = np.cos(beta)
    #     s_gamma = np.sin(gamma)
    #     c_gamma = np.cos(gamma)
    #     s_psi = np.sin(psi)
    #     c_psi = np.cos(psi)
    #     c_theta = np.cos(theta)
    #     s_theta = np.sin(theta)
    #     r = R_e + h
    #     m = w / g_0
    #     g = mu / r ** 2
    #
    #     J['hdot', 'v'] = s_gamma
    #     J['hdot', 'gamma'] = v * c_gamma
    #
    #     J['gammadot', 'lift'] = c_beta / (m * v)
    #     J['gammadot', 'h'] = c_gamma * (-v / r ** 2 + 2 * mu / (r ** 3 * v))
    #     J['gammadot', 'beta'] = -lift / (m * v) * s_beta
    #     J['gammadot', 'gamma'] = -s_gamma * (v / r - g / v)
    #     J['gammadot', 'v'] = -lift / (m * v ** 2) * c_beta + c_gamma * (1 / r + g / v ** 2)
    #
    #     J['phidot', 'v'] = c_gamma * s_psi / (c_theta * r)
    #     J['phidot', 'h'] = -v / r ** 2 * c_gamma * s_psi / c_theta
    #     J['phidot', 'gamma'] = -v / r * s_gamma * s_psi / c_theta
    #     J['phidot', 'psi'] = v / r * c_gamma * c_psi / c_theta
    #     J['phidot', 'theta'] = v / r * c_gamma * s_psi / (c_theta ** 2) * s_theta
    #
    #     J['psidot', 'v'] = -lift * s_beta / (m * c_gamma * v ** 2) + \
    #         c_gamma * s_psi * s_theta / (r * c_theta)
    #     J['psidot', 'gamma'] = lift * s_beta / (m * v * c_gamma ** 2) * s_gamma - \
    #         v * s_gamma * s_psi * s_theta / (r * c_theta)
    #     J['psidot', 'h'] = -v * c_gamma * s_psi * s_theta / (c_theta * r ** 2)
    #     J['psidot', 'beta'] = lift * c_beta / (m * v * c_gamma)
    #     J['psidot', 'theta'] = v * c_gamma * s_psi / (r * c_theta ** 2)
    #     J['psidot', 'psi'] = v * c_gamma * c_psi * s_theta / (r * c_theta)
    #     J['psidot', 'lift'] = s_beta / (m * v * c_gamma)
    #
    #     J['thetadot', 'v'] = c_gamma * c_psi / r
    #     J['thetadot', 'h'] = -v / r ** 2 * c_gamma * c_psi
    #     J['thetadot', 'gamma'] = -v / r * s_gamma * c_psi
    #     J['thetadot', 'psi'] = -v / r * c_gamma * s_psi
    #
    #     J['vdot', 'h'] = 2 * s_gamma * mu / r ** 3
    #     J['vdot', 'drag'] = -1 / m
    #     J['vdot', 'gamma'] = -g * c_gamma
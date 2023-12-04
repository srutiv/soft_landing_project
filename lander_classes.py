import openmdao.api as om
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
                           promotes_inputs=['beta', 'gamma', 'h', 'psi', 'theta', 'v', 'lift',
                                            'drag'],
                           promotes_outputs=['hdot', 'gammadot', 'phidot', 'psidot', 'thetadot',
                                             'vdot'])

################################USE THIS INSTEAD OR PUT THIS IN THE LANDERODE CLASS??????????"""
class FlightDynamics(ExplicitComponent):
    """
    Defines the flight dynamics for the shuttle reentry problem.

    References
    ----------
    .. [1] REPLACE WITH PROPER CITATION
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ######################## MIGHT HAVE TO SPLIT UP R AND RDOT #################################
        self.add_input('x', val=np.ones(nn), desc='x position', units='m')
        self.add_input('y', val=np.ones(nn), desc='y position', units='m')
        self.add_input('z', val=np.ones(nn), desc='z position', units='m')
        self.add_input('v_x', val=np.ones(nn), desc='x velocity', units='m/s')
        self.add_input('v_y', val=np.ones(nn), desc='y velocity', units='m/s')
        self.add_input('v_z', val=np.ones(nn), desc='z velocity', units='m/s')
        self.add_input('m', val=np.ones(nn), desc='mass', units='kg')
        self.add_input('Tc', val=np.ones(nn), desc='Thrust control', units='N')
        self.add_input('Gamma', val=np.ones(nn), desc='Thrust control Bound', units='N') #

        ######################## THE TIME DERIVATIVES ###############
        self.add_output('xdot', val=np.ones(nn), desc='x position rate', units='m/s')
        self.add_output('ydot', val=np.ones(nn), desc='y position rate', units='m/s')
        self.add_output('zdot', val=np.ones(nn), desc='z position rate', units='m/s')
        self.add_output('v_xdot', val=np.ones(nn), desc='x velocity rate', units='m/s**2')
        self.add_output('v_ydot', val=np.ones(nn), desc='y velocity rate', units='m/s**2')
        self.add_output('v_zdot', val=np.ones(nn), desc='z velocity rate', units='m/s**2')
        self.add_output('mDot', val=np.ones(nn), desc='mass change rate', units='kg/s')
        
        ####################### Time derivatives dont know if we need these? ####################################
        self.add_output('Tc', val=np.ones(nn), desc='Tc optimal', units='N')
        self.add_output('Gamma', val=np.ones(nn), desc='norm Tc optimal', units='N')
        self.add_output('q', val=np.ones(nn), desc='final landing coordinates', units='m') #### MAYBE SHOULD BE A PARAMETER
        
        ######################### May need to add omega as a parameter ############################ actually just put it in computer????

        partial_range = np.arange(nn, dtype=int)

        ################## I DONT THINK THESE SHOULD BE WITH RESPECT TO T
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
        r = inputs['r']
        rdot = inputs['rdot']
        m = inputs['m']
        Tc = inputs["Tc"]
        omega = inputs['omega']

        # constants
        Tmax = 24000 #N
        rho1 = 0.2 * Tmax  # thrust limit for throttle level 20%
        rho2 = 0.8 * Tmax
        alpha = 5e-4,  # s/m

        x = np.zeros((6,))
        x[0:3] = r
        x[3:6] = rdot

        g = np.ones((3,))  # constant gravity vector
        S = S_func(omega)

        A = A_func(omega)  # 6x6
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        xdot = np.dot(A, x) + np.dot(B, g + Tc / m)  # 6x1
        mdot = alpha * np.linalg.norm(Tc)
        zdot = np.append(xdot, mdot)

        outputs["rdot"] = xdot[0:3]
        outputs["rddot"] = xdot[3:6]
        outputs["mdot"] = mdot

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
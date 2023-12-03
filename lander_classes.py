import openmdao.api as om
import numpy as np
from openmdao.api import Group
from openmdao.api import ExplicitComponent, Problem

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

        self.add_output('r', val=np.ones(nn), desc='position', units='m')
        self.add_output('rdot', val=np.ones(nn), desc='velocity', units='m/s')
        self.add_output('tf', val=np.ones(nn), desc='tf optimal', units='s')
        self.add_output('Tc', val=np.ones(nn), desc='Tc optimal', units='N')
        self.add_output('Gamma', val=np.ones(nn), desc='norm Tc optimal', units='N')
        self.add_output('q', val=np.ones(nn), desc='final landing coordinates', units='m')

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials('hdot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('hdot', 'gamma', rows=partial_range, cols=partial_range)

        self.declare_partials('gammadot', 'lift', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'beta', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'v', rows=partial_range, cols=partial_range)

        self.declare_partials('phidot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'psi', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'theta', rows=partial_range, cols=partial_range)

        self.declare_partials('psidot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'beta', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'theta', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'psi', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'lift', rows=partial_range, cols=partial_range)

        self.declare_partials('thetadot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'psi', rows=partial_range, cols=partial_range)

        self.declare_partials('vdot', 'drag', rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'h', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        v = inputs['v']
        gamma = inputs['gamma']
        theta = inputs['theta']
        lift = inputs['lift']
        drag = inputs['drag']
        h = inputs['h']
        beta = inputs['beta']
        psi = inputs['psi']
        g_0 = 32.174
        w = 203000
        R_e = 20902900
        mu = .14076539e17
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        s_gamma = np.sin(gamma)
        c_gamma = np.cos(gamma)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = R_e + h
        m = w / g_0
        g = mu / r ** 2

        outputs['hdot'] = v * s_gamma
        outputs['gammadot'] = lift / (m * v) * c_beta + c_gamma * (v / r - g / v)
        outputs['phidot'] = v / r * c_gamma * s_psi / c_theta
        outputs['psidot'] = lift * s_beta / (m * v * c_gamma) + \
            v * c_gamma * s_psi * s_theta / (r * c_theta)
        outputs['thetadot'] = c_gamma * c_psi * v / r
        outputs['vdot'] = -drag / m - g * s_gamma

    def compute_partials(self, inputs, J):
        v = inputs['v']
        gamma = inputs['gamma']
        theta = inputs['theta']
        lift = inputs['lift']
        h = inputs['h']
        beta = inputs['beta']
        psi = inputs['psi']
        g_0 = 32.174
        w = 203000
        R_e = 20902900
        mu = .14076539e17
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        s_gamma = np.sin(gamma)
        c_gamma = np.cos(gamma)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = R_e + h
        m = w / g_0
        g = mu / r ** 2

        J['hdot', 'v'] = s_gamma
        J['hdot', 'gamma'] = v * c_gamma

        J['gammadot', 'lift'] = c_beta / (m * v)
        J['gammadot', 'h'] = c_gamma * (-v / r ** 2 + 2 * mu / (r ** 3 * v))
        J['gammadot', 'beta'] = -lift / (m * v) * s_beta
        J['gammadot', 'gamma'] = -s_gamma * (v / r - g / v)
        J['gammadot', 'v'] = -lift / (m * v ** 2) * c_beta + c_gamma * (1 / r + g / v ** 2)

        J['phidot', 'v'] = c_gamma * s_psi / (c_theta * r)
        J['phidot', 'h'] = -v / r ** 2 * c_gamma * s_psi / c_theta
        J['phidot', 'gamma'] = -v / r * s_gamma * s_psi / c_theta
        J['phidot', 'psi'] = v / r * c_gamma * c_psi / c_theta
        J['phidot', 'theta'] = v / r * c_gamma * s_psi / (c_theta ** 2) * s_theta

        J['psidot', 'v'] = -lift * s_beta / (m * c_gamma * v ** 2) + \
            c_gamma * s_psi * s_theta / (r * c_theta)
        J['psidot', 'gamma'] = lift * s_beta / (m * v * c_gamma ** 2) * s_gamma - \
            v * s_gamma * s_psi * s_theta / (r * c_theta)
        J['psidot', 'h'] = -v * c_gamma * s_psi * s_theta / (c_theta * r ** 2)
        J['psidot', 'beta'] = lift * c_beta / (m * v * c_gamma)
        J['psidot', 'theta'] = v * c_gamma * s_psi / (r * c_theta ** 2)
        J['psidot', 'psi'] = v * c_gamma * c_psi * s_theta / (r * c_theta)
        J['psidot', 'lift'] = s_beta / (m * v * c_gamma)

        J['thetadot', 'v'] = c_gamma * c_psi / r
        J['thetadot', 'h'] = -v / r ** 2 * c_gamma * c_psi
        J['thetadot', 'gamma'] = -v / r * s_gamma * c_psi
        J['thetadot', 'psi'] = -v / r * c_gamma * s_psi

        J['vdot', 'h'] = 2 * s_gamma * mu / r ** 3
        J['vdot', 'drag'] = -1 / m
        J['vdot', 'gamma'] = -g * c_gamma
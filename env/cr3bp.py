import numpy as np
import scipy.integrate as sci
from scipy.optimize import fsolve


class Parameters(object):
    pass


const = Parameters()

const.G = 6.67430e-20  # km^3 / kg /s -  universal gravitational constant
const.c = 299792.458  # km/s      - speed of light
const.au = 149597870.700  # km    - astronomical unit

# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430_and_de431.pdf - page 49, table 8
const.Sun_GM = 132712440041.939400
const.Mercury_GM = 22031.780000
const.Venus_GM = 324858.592000
const.Earth_GM = 398600.435436
const.Mars_GM = 42828.375214
const.Jupiter_GM = 126712764.800000
const.Saturn_GM = 37940585.200000
const.Uranus_GM = 5794548.600000
const.Neptune_GM = 6836527.100580
const.Pluto_GM = 977.000000
const.Moon_GM = 4902.800066


class CR3BP(object):

    def __init__(self, type):
        if type == 'EM':  # Earth-Moon
            self.Mp = const.Earth_GM / const.G  # Earth   index "p" stands for the primary body
            self.Ms = const.Moon_GM / const.G  # Moon    index "s" stands for the secondary body
            self.a = 384399.014  # [km] mean SMA
            self.primary_name = 'Earth'
            self.secondary_name = 'Moon'
        elif type == 'SE':  # Sun-Earth
            self.Mp = const.Sun_GM / const.G  # Sun
            self.Ms = const.Earth_GM / const.G  # Earth
            self.a = const.au  # [km] mean SMA
            self.primary_name = 'Sun'
            self.secondary_name = 'Earth'
        else:
            raise Exception('Unknown 3BP configuration')

        self.system_name = self.primary_name + '-' + self.secondary_name
        self.mu = self.Ms / (self.Ms + self.Mp)
        self.Rs = self.a * (1. - self.mu)
        self.Rp = self.a * self.mu
        self.mean_motion = np.sqrt(const.G * (self.Ms + self.Mp) / (self.a)**3)

    def get_L1_quintic(self):
        def quintic_L1(x):
            return x**5 - (3. - self.mu) * x**4 + (
                    3. - 2. * self.mu) * x**3 - self.mu * x**2 + 2. * self.mu * x - self.mu

        root = fsolve(quintic_L1, (1. - self.mu) / 2., xtol=1e-10)
        return root[0]

    def get_L2_quintic(self):
        def quintic_L2(x):
            return x - (1. - self.mu) / (x + self.mu) ** 2 - self.mu / (x - 1. + self.mu) ** 2

        root = fsolve(quintic_L2, 1 + self.mu, xtol=1e-10)
        return root[0]

    def potential(self, x, y, z):
        r10 = [x + self.mu, y, z]
        r20 = [x + self.mu - 1, y, z]
        r1 = np.linalg.norm(r10)
        r2 = np.linalg.norm(r20)

        return -0.5 * ((1 - self.mu) * (r1**2) + self.mu * (r2**2)) - (1 - self.mu) / r1 - self.mu / r2

    def get_jacobi_const(self, t, state):
        U = np.zeros(np.size(t))
        C = np.zeros(np.size(t))
        rel_error = np.zeros(np.size(t))

        for i in range(np.size(t)):
            C[i] = np.linalg.norm(state[i, 3:])**2 - 2 * self.potential(state[i, 0], state[i, 1], state[i, 2])

        rel_error = (C - C[0]) / C[0]

        return C, rel_error

    def rhs(self, t, state):
        x = state[0]
        y = state[1]
        z = state[2]
        v_x = state[3]
        v_y = state[4]
        v_z = state[5]

        r10 = [x + self.mu, y, z]
        r20 = [x + self.mu - 1., y, z]

        r1 = np.linalg.norm(r10)
        r2 = np.linalg.norm(r20)

        r13 = r1 ** 3
        r23 = r2 ** 3

        dxdt = np.zeros(6)
        dxdt[:3] = [v_x, v_y, v_z]
        dxdt[3] = 2 * v_y + x - (1 - self.mu) * (x + self.mu) / r13 - self.mu * (x - 1 + self.mu) / r23
        dxdt[4] = -2 * v_x + y - (1 - self.mu) * y / r13 - self.mu * y / r23
        dxdt[5] = -(1 - self.mu) * z / r13 - self.mu * z / r23

        return dxdt

    def integrate(self, x0, t_span):
        t_start = t_span[0]
        t_end = t_span[-1]

        solution = sci.solve_ivp(self.rhs, (t_start, t_end), x0,
                                 method='DOP853', t_eval=t_span, atol=1e-10, rtol=1e-10)

        return solution.y.T

    def halo_R3OA(self, Az, point, n_phase=1):
        def lambda_eqw(lmb):
            return lmb**4 + (c2 - 2.) * lmb**2 - (c2 - 1.) * (1. + 2. * c2)

        # Get L points
        n1 = np.sqrt(1 / self.a ** 3)  # tbp.mean_motion

        if point == "L1":
            gamma_L = (1. - self.mu) - self.get_L1_quintic()
            gamma_expr = gamma_L / (1. - gamma_L)
            c2 = (self.mu + (1. - self.mu) * gamma_expr**3) / gamma_L**3
            c3 = (self.mu - (1. - self.mu) * gamma_expr**4) / gamma_L**3
            c4 = (self.mu + (1. - self.mu) * gamma_expr**5) / gamma_L**3
        elif point == "L2":
            gamma_L = self.get_L2_quintic() - (1. - self.mu)
            gamma_expr = gamma_L / (1. + gamma_L)
            c2 = (self.mu + (1. - self.mu) * gamma_expr**3) / gamma_L**3
            c3 = (-self.mu - (1. - self.mu) * gamma_expr**4) / gamma_L**3
            c4 = (self.mu + (1. - self.mu) * gamma_expr**5) / gamma_L**3

        unit_length = gamma_L * self.a

        root = fsolve(lambda_eqw, 1000)
        lmbda = root[(np.imag(root) == 0 and root > 0)][0]

        k = 2. * lmbda / (lmbda**2 + 1. - c2)

        d1 = 3. * lmbda**2 * (k * (6. * lmbda**2 - 1.) - 2. * lmbda) / k
        d2 = 8. * lmbda**2 * (k * (11. * lmbda**2 - 1.) - 2. * lmbda) / k

        a21 = 3. * c3 * (k**2 - 2) / 4. / (1. + 2. * c2)
        a22 = 3. * c3 / 4. / (1. + 2 * c2)
        a23 = - 3. * c3 * lmbda * (3. * k ** 3 * lmbda - 6. * k * (k - lmbda) + 4.) / (4. * k * d1)
        a24 = - 3. * c3 * lmbda * (2. + 3. * k * lmbda) / (4. * k * d1)

        d21 = -c3 / 2. / lmbda**2
        d31 = 3. * (4. * c3 * a24 + c4) / 64. / lmbda**2
        d32 = 3. * (4. * c3 * (a23 - d21) + c4 * (4. + k ** 2)) / 64. / lmbda**2

        b21 = -3. * c3 * lmbda * (3. * k * lmbda - 4.) / 2. / d1
        b22 = 3. * c3 * lmbda / d1
        b31 = 3. * (8. * lmbda * (3. * c3 * (k * b21 - 2. * a23) - c4 * (2. + 3. * k**2)) +
                   (9. * lmbda**2 + 1. + 2. * c2) * (4. * c3 * (k * a23 - b21) + k * c4 * (4. + k**2))) / 8. / d2
        b32 = (9. * lmbda * (c3 * (k * b22 + d21 - 2. * a24) - c4) +
              3. * (9. * lmbda**2 + 1. + 2. * c2) * (4. * c3 * (k * a24 - b22) + k * c4) / 8.) / d2

        a31 = -9. * lmbda * (4. * c3 * (k * a23 - b21) + k * c4 * (4. + k**2)) / 4. / d2 + \
              (9. * lmbda**2 + 1. - c2) * (3. * c3 * (2. * a23 - k * b21) + c4 * (2. + 3. * k**2)) / 2. / d2
        a32 = -(9. * lmbda * (4. * c3 * (k * a24 - b22) + k * c4) / 4. +
              1.5 * (9. * lmbda**2 + 1. - c2) * (c3 * (k * b22 + d21 - 2. * a24) - c4)) / d2

        denom = 2. * lmbda * (lmbda * (1. + k**2) - 2. * k)
        s1 = (1.5 * c3 * (2. * a21 * (k ** 2 - 2.) - a23 * (k**2 + 2.) - 2. * k * b21) -
            3. * c4 * (3. * k**4 - 8 * k**2 + 8.) / 8.) / denom
        s2 = (1.5 * c3 * (2. * a22 * (k**2 - 2.) + a24 * (k**2 + 2.) + 2. * k * b22 + 5. * d21) +
            3. * c4 * (12. - k**2) / 8.) / denom

        a1 = -1.5 * c3 * (2. * a21 + a23 + 5. * d21) - 3. * c4 * (12. - k**2) / 8.
        a2 = 1.5 * c3 * (a24 - 2. * a22) + 9. * c4 / 8.

        l1 = a1 + 2. * lmbda**2 * s1
        l2 = a2 + 2. * lmbda**2 * s2

        Delta = lmbda**2 - c2

        Az /= unit_length

        Ax = np.sqrt((-l2 * Az**2 - Delta) / l1)
        Ay = k * Ax

        omega = 1 + s1 * Ax**2 + s2 * Az**2
        period = 2 * np.pi / omega

        theta = 0.
        delta_n = 2. - n_phase  # n_phase can be 1 or 3 (input argument)
        tau_1 = 0. + theta  # tau_z = 0. + (theta + np.pi * n / 2) this is taken care of by delta_n

        x0 = a21 * Ax**2 + a22 * Az**2 - Ax * np.cos(tau_1) + \
            (a23 * Ax**2 - a24 * Az**2) * np.cos(2. * tau_1) + \
            (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3. * tau_1)
        z0 = delta_n * (Az * np.cos(tau_1) + d21 * Ax * Az * (np.cos(2. * tau_1) - 3.) +
                        (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3. * tau_1))
        vy0 = k * Ax * np.cos(tau_1) + \
            (b21 * Ax**2 - b22 * Az**2) * (2. * np.cos(2. * tau_1)) + \
            (b31 * Ax**3 - b32 * Ax * Az**2) * 3. * np.cos(3. * tau_1)

        richardson = Parameters()
        richardson.gamma_L = gamma_L
        richardson.lmbda = lmbda
        richardson.k = k
        richardson.Delta = Delta
        richardson.c2 = c2
        richardson.c3 = c3
        richardson.c4 = c4
        richardson.s1 = s1
        richardson.s2 = s2
        richardson.l1 = l1
        richardson.l2 = l2
        richardson.a1 = a1
        richardson.a2 = a2
        richardson.d1 = d1
        richardson.d2 = d2
        richardson.a21 = a21
        richardson.a22 = a22
        richardson.a23 = a23
        richardson.a24 = a24
        richardson.a31 = a31
        richardson.a32 = a32
        richardson.b21 = b21
        richardson.b22 = b22
        richardson.b31 = b31
        richardson.b32 = b32
        richardson.d21 = d21
        richardson.d31 = d31
        richardson.d32 = d32
        richardson.Ax = Ax
        richardson.Az = Az
        richardson.delta_n = delta_n

        return np.array([x0, 0, z0, 0, vy0, 0]), richardson, period


# Richardson's 3rd order approximation trajectory given the time array tau_1,
# a set of Richardson's constants, and a delta_n (\pm 1)
def richardson_traj(tau_1, cnst, delta_n=1):
    X = np.zeros((np.size(tau_1), 3))
    tau_1 = tau_1 * cnst.lmbda

    X[:, 0] = cnst.a21 * cnst.Ax**2 + cnst.a22 * cnst.Az**2 - cnst.Ax * np.cos(tau_1) + \
              (cnst.a23 * cnst.Ax**2 - cnst.a24 * cnst.Az**2) * np.cos(2. * tau_1) + \
              (cnst.a31 * cnst.Ax**3 - cnst.a32 * cnst.Ax * cnst.Az**2) * np.cos(3. * tau_1)

    X[:, 1] = cnst.k * cnst.Ax * (np.sin(tau_1)) + \
              (cnst.b21 * cnst.Ax**2 - cnst.b22 * cnst.Az**2) * np.sin(2. * tau_1) + \
              (cnst.b31 * cnst.Ax**3 - cnst.b32 * cnst.Ax * cnst.Az**2) * np.sin(3. * tau_1)

    X[:, 2] = delta_n * (cnst.Az * np.cos(tau_1) + cnst.d21 * cnst.Ax * cnst.Az * (np.cos(2. * tau_1) - 3.) +
                         (cnst.d32 * cnst.Az * cnst.Ax**2 - cnst.d31 * cnst.Az**3) * np.cos(3. * tau_1))

    return X


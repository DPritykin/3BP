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

    def __init__(self, system_name):
        if system_name == 'EM':  # Earth-Moon
            self.Mp = const.Earth_GM / const.G  # Earth   index "p" stands for the primary body
            self.Ms = const.Moon_GM / const.G  # Moon    index "s" stands for the secondary body
            self.a = 384399.014  # [km] mean SMA
            self.primary_name = 'Earth'
            self.secondary_name = 'Moon'
        elif system_name == 'SE':  # Sun-Earth
            self.Mp = const.Sun_GM / const.G  # Sun
            self.Ms = const.Earth_GM / const.G  # Earth
            self.a = const.au  # [km] mean SMA
            self.primary_name = 'Sun'
            self.secondary_name = 'Earth'
        else:
            raise Exception('Unsupported 3BP system')

        self.system_name = self.primary_name + '-' + self.secondary_name
        self.mu = self.Ms / (self.Ms + self.Mp)
        self.Rs = self.a * (1. - self.mu)
        self.Rp = self.a * self.mu
        self.mean_motion = np.sqrt(const.G * (self.Ms + self.Mp) / self.a**3)

    def get_l1_quintic(self):
        def quintic_l1(x):
            return x**5 - (3. - self.mu) * x**4 + (3. - 2. * self.mu) * x**3 - \
                   self.mu * x**2 + 2. * self.mu * x - self.mu

        root = fsolve(quintic_l1, np.array((1. - self.mu) / 2.), xtol=1e-10)
        return root[0]

    def get_l2_quintic(self):
        def quintic_l2(x):
            return x - (1. - self.mu) / (x + self.mu)**2 - self.mu / (x - 1. + self.mu)**2

        root = fsolve(quintic_l2, np.array(1 + self.mu), xtol=1e-10)
        return root[0]

    def potential(self, x, y, z):
        r10 = [x + self.mu, y, z]
        r20 = [x + self.mu - 1, y, z]
        r1 = np.linalg.norm(r10)
        r2 = np.linalg.norm(r20)

        return -0.5 * ((1 - self.mu) * (r1**2) + self.mu * (r2**2)) - (1 - self.mu) / r1 - self.mu / r2

    def get_jacobi_const(self, t, state):
        c = np.zeros(np.size(t))

        for i in range(np.size(t)):
            c[i] = np.linalg.norm(state[i, 3:])**2 - 2 * self.potential(state[i, 0], state[i, 1], state[i, 2])

        rel_error = (c - c[0]) / c[0]

        return c, rel_error

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

        r13 = r1**3
        r23 = r2**3

        dxdt = np.zeros(6)
        dxdt[:3] = [v_x, v_y, v_z]
        dxdt[3] = 2 * v_y + x - (1 - self.mu) * (x + self.mu) / r13 - self.mu * (x - 1 + self.mu) / r23
        dxdt[4] = -2 * v_x + y - (1 - self.mu) * y / r13 - self.mu * y / r23
        dxdt[5] = -(1 - self.mu) * z / r13 - self.mu * z / r23

        return dxdt

    def stm_rhs(self, t, stm_state):
        dxdt = np.zeros(42)
        dxdt[0:6] = self.rhs(t, stm_state[0:6])

        a_stm = self.cr3bp_jacobian(stm_state[0:6])
        stm_dot = a_stm.dot(np.reshape(stm_state[6:], (6, 6)))
        dxdt[6:] = np.reshape(stm_dot, 36)

        return dxdt

    def xz_crossing_event(self, t, stm_state):
        return stm_state[1]

    def integrate(self, x0, t_span):
        t_start = t_span[0]
        t_end = t_span[-1]

        solution = sci.solve_ivp(self.rhs, (t_start, t_end), x0,
                                 method='DOP853', t_eval=t_span, atol=1e-10, rtol=1e-10)
        return solution.y.T

    def stm_integrate(self, x0, t_span):
        init_cond = np.hstack((np.reshape(x0, (1, 6)), np.reshape(np.eye(6), (1, 36))))

        cross_xz_event = lambda t, x: self.xz_crossing_event(t, x)
        cross_xz_event.terminal = True
        cross_xz_event.direction = -1 if x0[4] > 0 else 1
        solution = sci.solve_ivp(self.stm_rhs, t_span, init_cond.squeeze(),
                                 method='DOP853', events=cross_xz_event, atol=1e-12, rtol=1e-12)
        state_end = np.squeeze(solution.y_events[0])
        time_end = solution.t_events[0][0]
        stm = np.reshape(state_end[6:], (6, 6))

        return time_end, state_end[0:6], stm

    def halo_xzydot_diff_corrector(self, state, period, eps=1e-8, maxiter=10):
        initial_state = state
        count = 1

        while True:
            if count > maxiter:
                print('Maximum iterations reached in differential correction')
                break
            count += 1

            # computing the state transition matrix at half period
            xz_crossing_time, half_period_state, stm = self.stm_integrate(initial_state, [0, period])
            period = 2 * xz_crossing_time

            # correction for x_dot, z_dot (see Howell, 1984)
            dx_dot_dz_dot = -np.array([half_period_state[3], half_period_state[5]])
            if np.linalg.norm(dx_dot_dz_dot) < eps:
                break

            dxdt = self.rhs(xz_crossing_time, half_period_state)
            x_ddot = dxdt[3]
            z_ddot = dxdt[5]
            y_dot = dxdt[1]
            corr_mat = np.array([[stm[3, 2], stm[3, 4]], [stm[5, 2], stm[5, 4]]])
            corr_mat[0, 0] -= x_ddot * stm[1, 2] / y_dot
            corr_mat[0, 1] -= x_ddot * stm[1, 4] / y_dot
            corr_mat[1, 0] -= z_ddot * stm[1, 2] / y_dot
            corr_mat[1, 1] -= z_ddot * stm[1, 4] / y_dot

            state_corr = np.linalg.solve(corr_mat, dx_dot_dz_dot)
            initial_state[2] += state_corr[0]
            initial_state[4] += state_corr[1]

        return initial_state

    def halo_r3oa(self, Az_km, lp_name, n_phase=1):
        def lambda_eqw(lmb):
            return lmb**4 + (c2 - 2.) * lmb**2 - (c2 - 1.) * (1. + 2. * c2)

        # Get L points
        if lp_name == "L1":
            gamma_l = (1. - self.mu) - self.get_l1_quintic()
            gamma_expr = gamma_l / (1. - gamma_l)
            c2 = (self.mu + (1. - self.mu) * gamma_expr**3) / gamma_l**3
            c3 = (self.mu - (1. - self.mu) * gamma_expr**4) / gamma_l**3
            c4 = (self.mu + (1. - self.mu) * gamma_expr**5) / gamma_l**3
        elif lp_name == "L2":
            gamma_l = self.get_l2_quintic() - (1. - self.mu)
            gamma_expr = gamma_l / (1. + gamma_l)
            c2 = (self.mu + (1. - self.mu) * gamma_expr**3) / gamma_l**3
            c3 = (-self.mu - (1. - self.mu) * gamma_expr**4) / gamma_l**3
            c4 = (self.mu + (1. - self.mu) * gamma_expr**5) / gamma_l**3
        else:
            raise Exception('Unsupported libration point specification')

        unit_length = gamma_l * self.a

        root = fsolve(lambda_eqw, np.array(1000))
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

        Az = Az_km / unit_length

        Ax = np.sqrt((-l2 * Az**2 - Delta) / l1)
        Ay = k * Ax

        omega = 1 + s1 * Ax**2 + s2 * Az**2
        period = 2 * np.pi / omega / lmbda

        theta = 0.
        delta_n = 2. - n_phase  # n_phase can be 1 or 3 (input argument)
        tau_1 = 0. + theta  # tau_z = 0. + (theta + np.pi * n / 2) this is taken care of by delta_n

        x0 = a21 * Ax**2 + a22 * Az**2 - Ax * np.cos(tau_1) + \
            (a23 * Ax**2 - a24 * Az**2) * np.cos(2. * tau_1) + \
            (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3. * tau_1)
        z0 = delta_n * (Az * np.cos(tau_1) + d21 * Ax * Az * (np.cos(2. * tau_1) - 3.) +
                        (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3. * tau_1))
        vy0 = (k * Ax * np.cos(tau_1) + \
            (b21 * Ax**2 - b22 * Az**2) * (2. * np.cos(2. * tau_1)) + \
            (b31 * Ax**3 - b32 * Ax * Az**2) * 3. * np.cos(3. * tau_1))

        richardson = Parameters()
        richardson.gamma_l = gamma_l
        richardson.lmbda = lmbda
        richardson.omega = omega
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

    def richardson2barycentric(self, state, cnst):
        state *= np.abs(cnst.gamma_l)
        if state.ndim > 1:
            state[:, 0] += (1 - self.mu + cnst.gamma_l)
            state[:, 3:] *= (cnst.lmbda * cnst.omega)
        else:
            state[0] += (1 - self.mu + cnst.gamma_l)
            state[3:] *= (cnst.lmbda * cnst.omega)

        return state

    def cr3bp_jacobian(self, state):
        x = state[0]
        y = state[1]
        z = state[2]

        r10 = [x + self.mu, y, z]
        r20 = [x + self.mu - 1., y, z]
        r = [x, y, z]

        r1 = np.linalg.norm(r10)
        r2 = np.linalg.norm(r20)

        r_13 = r1**(-3)  # 1/r1^3
        r_23 = r2**(-3)  # 1/r2^3
        r_15 = r1**(-5)  # 1/r1^5
        r_25 = r2**(-5)  # 1/r2^5

        u_bar2 = np.empty((3, 3))

        u_bar2[0, 0] = self.mu * r_23 + (1.0 - self.mu) * r_13 - \
                       3.0 * self.mu * (state[0] - 1.0 + self.mu) * (state[0] - 1.0 + self.mu) * r_25 - \
                       3.0 * (state[0] + self.mu) * (state[0] + self.mu) * (1.0 - self.mu) * r_15 - 1.0

        u_bar2[0, 1] = 3.0 * state[1] * (self.mu + state[0]) * (self.mu - 1.0) * r_15 - \
                       3.0 * self.mu * state[1] * (state[0] - 1.0 + self.mu) * r_25

        u_bar2[0, 2] = 3.0 * state[2] * (self.mu + state[0]) * (self.mu - 1.0) * r_15 - \
                       3.0 * self.mu * state[2] * (state[0] - 1.0 + self.mu) * r_25

        u_bar2[1, 0] = u_bar2[0, 1]

        u_bar2[1, 1] = self.mu * r_23 - (self.mu - 1.0) * r_13 + \
                       3.0 * state[1] * state[1] * (self.mu - 1.0) * r_15 - \
                       3.0 * self.mu * state[1] ** 2 * r_25 - 1.0

        u_bar2[1, 2] = 3.0 * state[1] * state[2] * (self.mu - 1.0) * r_15 - \
                       3.0 * self.mu * state[1] * state[2] * r_25

        u_bar2[2, 0] = u_bar2[0, 2]
        u_bar2[2, 1] = u_bar2[1, 2]

        u_bar2[2, 2] = self.mu * r_23 - (self.mu - 1.0) * r_13 + \
                       3.0 * state[2] ** 2 * (self.mu - 1.0) * r_15 - 3.0 * self.mu * state[2] ** 2 * r_25

        a_stm = np.zeros((6, 6))
        a_stm[0:3, 3:6] = np.eye(3)
        a_stm[3:6, 0:3] = - u_bar2
        a_stm[3, 4] = 2.0
        a_stm[4, 3] = - 2.0

        return a_stm

    def stm_matrix(self, stm_state):
        stm = np.zeros((3, 3))
        return stm

# Richardson's 3rd order approximation trajectory given the time array tau_1,
# a set of Richardson's constants, and a delta_n (\pm 1)
def richardson_traj(tau_1, cnst, delta_n=1):
    r = np.zeros((np.size(tau_1), 3))
    tau_1 = tau_1 * cnst.lmbda * cnst.omega

    r[:, 0] = cnst.a21 * cnst.Ax**2 + cnst.a22 * cnst.Az**2 - cnst.Ax * np.cos(tau_1) + \
              (cnst.a23 * cnst.Ax**2 - cnst.a24 * cnst.Az**2) * np.cos(2. * tau_1) + \
              (cnst.a31 * cnst.Ax**3 - cnst.a32 * cnst.Ax * cnst.Az**2) * np.cos(3. * tau_1)

    r[:, 1] = cnst.k * cnst.Ax * (np.sin(tau_1)) + \
              (cnst.b21 * cnst.Ax**2 - cnst.b22 * cnst.Az**2) * np.sin(2. * tau_1) + \
              (cnst.b31 * cnst.Ax**3 - cnst.b32 * cnst.Ax * cnst.Az**2) * np.sin(3. * tau_1)

    r[:, 2] = delta_n * (cnst.Az * np.cos(tau_1) + cnst.d21 * cnst.Ax * cnst.Az * (np.cos(2. * tau_1) - 3.) +
                         (cnst.d32 * cnst.Az * cnst.Ax**2 - cnst.d31 * cnst.Az**3) * np.cos(3. * tau_1))

    return r

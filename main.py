import numpy as np
import env.cr3bp as cr3bp
import visualization.cr3bp_plots as plt


if __name__ == '__main__':
    system_name = 'EM'
    em3bp = cr3bp.CR3BP(system_name)
    print(f'The following constant (for the {em3bp.system_name} system) is used, mu = {em3bp.mu}')

    l2_x = em3bp.get_L2_quintic()
    print(f'The following L2 (for the {em3bp.system_name} system) distance is obtained, L2 = {l2_x}')

    # plotting solution of diff equations for the initial conditions at L2
    l2_eq_traj = em3bp.integrate([l2_x, 0., 0., 0., 0., 0.], np.arange(0., 10., 0.05))
    plt.plot_trajectory(l2_eq_traj * em3bp.a, 'Three Bodies in Space', em3bp)

    _, richardson_const, period = cr3bp.halo_R3OA(50000, system=system_name, point="L2")
    approx_traj = cr3bp.richardson_traj(np.arange(0., 1.5 * period, 0.01), richardson_const, 1.)

    approx_traj *= richardson_const.gamma_L
    approx_traj[:, 0] += (1 - em3bp.mu + richardson_const.gamma_L)
    approx_traj *= em3bp.a

    plt.plot_trajectory(approx_traj, 'Richardson Halo', em3bp)

    plt.mayavi_plot(approx_traj / em3bp.a)

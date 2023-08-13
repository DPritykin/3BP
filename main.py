import numpy as np
import env.cr3bp as cr3bp
import visualization.cr3bp_plots as plt


if __name__ == '__main__':
    em3bp = cr3bp.CR3BP('SE')
    print(f'The following constant (for the {em3bp.system_name} system) is used, mu = {em3bp.mu}')

    l2_x = em3bp.get_L2_quintic()
    print(f'The following L2 (for the {em3bp.system_name} system) distance is obtained, L2 = {l2_x}')

    # plotting solution of diff equations for the initial conditions at L2
    # l2_eq_traj = em3bp.integrate([l2_x, 0., 0., 0., 0., 0.], np.arange(0., 10., 0.05))
    # plt.plot_trajectory(l2_eq_traj, 'Three Bodies in Space', em3bp)

    _, richardson_const = cr3bp.halo_R3OA(125000, system="SE", point="L2")
    approx_traj = cr3bp.richardson_traj(np.arange(0., 3., 0.05), richardson_const)
    plt.plot_trajectory(approx_traj, 'Richardson Halo', em3bp)

    plt.mayavi_plot(approx_traj)

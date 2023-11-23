import numpy as np
import env.cr3bp as cr3bp
import visualization.cr3bp_plots as plt


if __name__ == '__main__':
    em3bp = cr3bp.CR3BP('EM')
    print(f'The following constant (for the {em3bp.system_name} system) is used, mu = {em3bp.mu}')

    l2_x = em3bp.get_l2_quintic()
    print(f'The following L2 (for the {em3bp.system_name} system) distance is obtained, L2 = {l2_x}')

    # plotting solution of diff equations for the initial conditions at L2
    l2_eq_traj = em3bp.integrate([l2_x, 0., 0., 0., 0., 0.], np.arange(0., 10., 0.05))

    # plotting the trajectory with initial conditions in L2 equilibrium
    # plt.plot_trajectory(l2_eq_traj * em3bp.a, 'Three Bodies in Space', em3bp)

    # computing Richardson 3rd order halo orbit approximation
    richardson_initial, richardson_const, period = em3bp.halo_r3oa(Az_km=50000, lp_name="L2")
    # and converting them to the barycentric frame
    first_guess_initials = em3bp.richardson2barycentric(richardson_initial, richardson_const)
    print(f'Initial conditions from Richardson 3rd order approximation, x0 = {first_guess_initials}')

    # plotting Richardson 3rd order halo orbit approximation
    # approx_traj_r = cr3bp.richardson_traj(np.arange(0., period, 0.01), richardson_const)
    # approx_traj_b = em3bp.richardson2barycentric(approx_traj_r, richardson_const)
    # plt.plot_trajectory(approx_traj_r[:, 0:3] * em3bp.a, 'Richardson Halo', em3bp)

    # plotting the trajectory propagated from Richardson 3rd order halo orbit initial conditions
    # richardson_initials_traj = em3bp.integrate(first_guess_initials, np.arange(0., period, 0.05))
    # plt.plot_trajectory(richardson_initials_traj[:, 0:3] * em3bp.a, 'Richardson Trajectory', em3bp)

    # applying diff corrector and plotting the resulting trajectory
    #corrected_initials = em3bp.halo_xzydot_diff_corrector(first_guess_initials, period)
    #print(f'Initial conditions after differential corrector, x0 = {corrected_initials}')
    #corrected_traj = em3bp.integrate(corrected_initials, np.arange(0., 2 * period, 0.0005))
    #plt.plot_trajectory(corrected_traj[:, 0:3] * em3bp.a, 'Diff Corrector Trajectory', em3bp)

    # butterfly orbit
    butterfly_initials = np.array([1.0354, 0., -0.1740, 0., -0.0801, 0.])
    period = 3
    butterfly_traj = em3bp.integrate(butterfly_initials, np.arange(0., period, 0.001))
    plt.plot_trajectory(butterfly_traj[:, 0:3] * em3bp.a, 'Butterfly Orbit', em3bp)

    # mayavi plot
    # plt.mayavi_plot(approx_traj_r)
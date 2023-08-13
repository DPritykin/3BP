import env.cr3bp as cr3bp
import matplotlib.pyplot as plt
import mayavi.mlab as mlb
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory(trajectory, caption, cr3bp_obj):
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(caption)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(trajectory[:, 0] * cr3bp_obj.a, trajectory[:, 1] * cr3bp_obj.a, trajectory[:, 2] * cr3bp_obj.a,
               color='r', marker='.', label='L2')
    ax.scatter(-cr3bp_obj.Rp, 0, 0, color='g', marker='o', label=cr3bp_obj.primary_name)
    ax.scatter(cr3bp_obj.Rs, 0, 0, color='b', marker='.', label=cr3bp_obj.secondary_name)
    ax.legend()
    plt.show()


def mayavi_plot(trajectory):
    mlb.plot3d(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    mlb.show()
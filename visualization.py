
from utils import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def drawColoredPath(x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
                    linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    # draw path
    fig, ax = plt.subplots(nrows=1, figsize=(20, 15))

    ax.add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    north_pole = cities[cities.CityId == 0]
    plt.scatter(north_pole.X, north_pole.Y, marker='*', c='red', s=1000)

    list_prime = [i for i in primerange(0, df_cities.shape[0] + 1)]
    coords = cities[cities.CityId.isin(list_prime)]
    plt.scatter(x=coords.X, y=coords.Y, marker='x', c='red', s=5, alpha=0.6)
    #plt.scatter(x=cities.X, y=cities.Y, c='red', s=3, alpha=0.6)

    # draw diagram color - index of step/city
    fig, ax = plt.subplots(nrows=1, figsize=(15, 0.5))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(x))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                     norm=norm,
                                     orientation='horizontal')
    ax.set_xlabel('Colors and steps mapping')
    ax.xaxis.set_label_position('top')

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_colored_path(path, penalization, type = 'pen'):
    coords = cities[['X', 'Y']].values
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)

    x, y = ordered_coords[:, 0], ordered_coords[:, 1]
    z = ((penalization > 0) +.3) #
    if type == 'lin':
        z = np.linspace(0, 1, len(x))
    drawColoredPath(x, y, z, cmap=plt.get_cmap('Greys'),
                    linewidth=1)


def calc_score_with_pen(list_path, df_cities):
    np_xy_cities = df_cities[['X', 'Y']].values
    list_path = list_path.tolist()
    len_path = len(list_path)
    # Calc Distance
    np_xy_path = np_xy_cities[list_path]
    np_dist_path = np.sum((np_xy_path[0:-1] - np_xy_path[1:]) ** 2, axis=1) ** 0.5
    # List of Primes 0 to (len_path-1)
    list_prime = [i for i in primerange(0, df_cities.shape[0] + 1)]
    # Flag np.array, is path's from-city number non-prime?
    np_is_non_prime = np.ones(len_path + 1)
    np_is_non_prime[list_prime] = 0
    np_is_non_prime = np_is_non_prime
    np_is_path_from_non_prime = np_is_non_prime[list_path][:-1]
    # Flag np.array, is path number(1 start) % 10 == 0?
    np_is_path_num_per_ten = np.array(([0] * 9 + [1]) * ((len_path - 1) // 10) + [0] * ((len_path - 1) % 10))
    # If both flags are true, *1.1, else * 1.0
    penalization = np_dist_path * (0.1 * np_is_path_from_non_prime * np_is_path_num_per_ten)
    dists = np_dist_path * (1.0 + 0.1 * np_is_path_from_non_prime * np_is_path_num_per_ten)

    print(np.sum(dists)-np.sum(penalization), np.sum(dists), np.sum(penalization), penalization[penalization>1].sum(), penalization[penalization>5].sum(), penalization.max())
    return np.sum(dists), np.sum(penalization), penalization[penalization>1].sum(), dists, penalization


score, _, _, dists, penalization = calc_score_with_pen(solution_neos, cities)

print('max_dist: ',np.max(dists))
print('mean_dist: ',np.mean(dists))
print('max_penalization: ',np.max(penalization))
print('mean_penalization: ',np.mean(penalization))

show_path = solution_neos
plot_colored_path(show_path, penalization, type = 'lin')

calc_score_with_pen(solution_neos, cities)

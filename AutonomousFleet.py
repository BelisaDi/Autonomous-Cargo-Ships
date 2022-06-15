import numpy as np
from scipy.spatial import KDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath


class AutonomousFleet:
    """
    Class that stores a fleet of ships as points on the 2D plane, using
    a KD-Tree.

    Attributes:
    ----------------
    T: KD-Tree that stores the vessels coordinates

    Public methods:
    ----------------
    nearest_ships(ship_id, s, plot = False): returns the s closest neighbors of
        the ship given by its ID ship_id, and plots if indicated
    avoid_collisions(ship_id, r, angle, plot = False): reports all the ships
        within a square of length r, with given angle of rotation, and center
        given by the location of the ship given by its ID ship_id, plots if
        indicated
    leading_ships(plot = False): report all the ships that have locations with
        either maximal or minimal x or y coordinates, plots if indicated
    plot(animate = False): plots the KD-Tree decomposition of the fleet. If
        indicated, will animate the recursive process
    """

    def __init__(self, data):
        """
        Parameters:
        -------------
        data: coordinates of the vessels in 2D
        """
        self.T = KDTree(data, leafsize=1)

    def nearest_ships(self, ship_id, s, plot=False):
        """
        Given the index of the ship, finds and reports the s closest ships
            using the Euclidean distance.

        Parameters
        --------------
        ship_id: index of the ship within the fleet
        s: required number of closest neighbors
        plot: boolean, if True, plots the ship in green, and its closest
            neighbors in red

        Output
        --------------
        ind: indexes of the closest ships within the fleet
        """
        ship = self.T.data[ship_id]
        dist, ind = self.T.query(ship, k=s + 1)
        ind = np.delete(ind, 0)
        neigh_coords = np.array(self.T.data[ind])

        if plot:
            plt.scatter(self.T.data[:, 0], self.T.data[:, 1])

            lab = ' '.join([str(elem) for elem in ind])
            plt.scatter(neigh_coords[:, 0], neigh_coords[:, 1], c="red",
                        label=lab)

            plt.scatter(ship[0], ship[1], c="green", label=ship_id)

            plt.legend(loc="best", fancybox=True,
                       title="Ships IDs")
            plt.ylabel('y')
            plt.xlabel('x')
            plt.title(str(s) + " closest neighbors of ship " + str(ship_id))
            plt.show()

        return ind

    def avoid_collisions(self, ship_id, r, angle, plot=False):
        """
        Finds and returns whether there are any ships within a square of given
            orientation of length r and center given by the location of the
            ship given by ship_id

        Parameters
        -------------
        ship_id: index of the ship within the fleet
        r: length of the square edge
        angle: desired rotation around the x axis
        plot: boolean, if True, plots the ship in green, the square, and the
            ships inside it in red

        Output
        -------------
        in_square_idx: indexes of the ships within the square
        """
        ship = self.T.data[ship_id]
        radius = abs(r) / np.sqrt(2)
        ind = self.T.query_ball_point(ship, radius)
        ind = np.delete(ind, ind.index(ship_id))

        origin_x = ship[0] - r / 2
        origin_y = ship[1] - r / 2

        fig, ax = plt.subplots()

        square = patches.Rectangle(
            (origin_x, origin_y), r, r, color="red", alpha=0.2)
        rotation = mpl.transforms.Affine2D().rotate_around(
            ship[0], ship[1], angle)
        square.set_transform(rotation + ax.transData)

        coords = square.get_patch_transform().transform(
            square.get_path().vertices[:-1])

        coords = rotation.transform(coords)
        poly = []

        m = len(coords)
        for i in range(m + 1):
            poly.append((coords[i % m, 0], coords[i % m, 1]))
        poly_path = mplPath.Path(poly)

        in_square_idx = []

        for i in ind:
            if poly_path.contains_point(self.T.data[i]):
                in_square_idx.append(i)

        point_coords = np.array(self.T.data[in_square_idx])

        if plot:
            plt.scatter(self.T.data[:, 0], self.T.data[:, 1])

            lab = ' '.join([str(elem) for elem in in_square_idx])
            plt.scatter(point_coords[:, 0], point_coords[:, 1], c="red",
                        label=lab)
            plt.scatter(ship[0], ship[1], c="green", label=ship_id)

            ax.add_patch(square)

            plt.legend(loc="best", fancybox=True,
                       title="Ships IDs")
            plt.ylabel('y')
            plt.xlabel('x')
            plt.title("Ships within the square of length " + str(r) +
                      " with angle of rotation " + str(angle) + "°")
            plt.show()

        return in_square_idx

    def leading_ships(self, plot=False):
        """
        Reports all the ships that have locations with either maximal or
            minimal x or y coordinates.

        Parameters
        -------------
        plot: boolean, if True, plots the ships with maximal or minimal x
            or y coordinates

        Output
        -------------
        north: indexes of ships with maximum y coordinate
        south: indexes of ships with minimum y coordinate
        east: indexes of ships with maximum x coordinate
        west: indexes of ships with minimum x coordinate
        """
        east = np.where(self.T.data[:, 0] == self.T.maxes[0])
        north = np.where(self.T.data[:, 1] == self.T.maxes[1])
        west = np.where(self.T.data[:, 0] == self.T.mins[0])
        south = np.where(self.T.data[:, 1] == self.T.mins[1])

        if plot:
            max_mins = np.concatenate((east, north, west, south), axis=None)
            u, c = np.unique(max_mins, return_counts=True)
            dups = u[c > 1]

            dups_direction = []
            if len(dups) != 0:
                for d in dups:
                    dirs = []
                    if d in east[0]:
                        dirs.append("E")
                    if d in north[0]:
                        dirs.append("N")
                    if d in west[0]:
                        dirs.append("W")
                    if d in south[0]:
                        dirs.append("S")
                    dups_direction.append([d, dirs])

            fig, ax = plt.subplots()
            plt.scatter(self.T.data[:, 0], self.T.data[:, 1], c="black", s=5)

            north_lab = "North: " + ' '.join([str(elem) for elem in north[0]])
            south_lab = "South: " + ' '.join([str(elem) for elem in south[0]])
            east_lab = "East: " + ' '.join([str(elem) for elem in east[0]])
            west_lab = "West: " + ' '.join([str(elem) for elem in west[0]])

            plt.scatter(np.array(self.T.data[north])[:, 0],
                        np.array(self.T.data[north])[:, 1], marker="^",
                        c="red", label=north_lab)

            plt.scatter(np.array(self.T.data[south])[:, 0],
                        np.array(self.T.data[south])[:, 1], marker="v",
                        c="red", label=south_lab)

            plt.scatter(np.array(self.T.data[east])[:, 0],
                        np.array(self.T.data[east])[:, 1], marker=">", c="red",
                        label=east_lab)

            plt.scatter(np.array(self.T.data[west])[:, 0],
                        np.array(self.T.data[west])[:, 1], marker="<", c="red",
                        label=west_lab)

            if len(dups_direction) != 0:
                for couple in dups_direction:
                    plt.scatter(self.T.data[couple[0], 0],
                                self.T.data[couple[0], 1], marker="s", c="red")
                    ax.annotate(' '.join([str(elem) for elem in couple[1]]),
                                (self.T.data[couple[0], 0],
                                self.T.data[couple[0], 1]),
                                weight='bold', fontsize=10)

            plt.legend(loc="best", fancybox=True,
                       title="Ships IDs")
            plt.ylabel('y')
            plt.xlabel('x')
            plt.title("Ships with maximum or minimum x or y coordinate")
            plt.show()

        return north[0], south[0], east[0], west[0]

    def plot(self, animate=False):
        """
        Generates a two-dimensional map with the location of each ship in
            the fleet. Represent ships as dots and the splitting lines

        Parameters
        ------------
        animate: boolen, if True, it will animate the process of the splitting
            lines. It's not recommended for large amounts of data, as it may
            be too slow
        """

        self._plot_helper(self.T.data, self.T.mins[0] - 0.2,
                          self.T.maxes[0] + 0.2, self.T.mins[1] - 0.2,
                          self.T.maxes[1] + 0.2, 0, None, None, animate)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title("KD-Tree diagram")
        plt.show()

    def _plot_helper(self, nodes, min_x, max_x, min_y, max_y, depth,
                     prev_node, is_left, is_animation):
        """
        Recursive function that does the actual plotting
        """

        if depth == 0:
            plt.scatter(nodes[:, 0], nodes[:, 1], c="black", zorder=2)

        if len(nodes) == 1:
            return

        ax = depth % 2
        nodes = nodes[nodes[:, ax].argsort()]

        med_idx = (len(nodes) // 2) - 1
        median = nodes[med_idx]

        left_branch = nodes[:med_idx + 1]
        right_branch = nodes[med_idx + 1:]

        if ax == 0:
            if is_left is not None and prev_node is not None:
                if is_left:
                    max_y = prev_node[1]
                else:
                    min_y = prev_node[1]

            plt.plot([median[0], median[0]], [min_y, max_y],
                     color="red", zorder=1)
            if is_animation:
                plt.pause(1)

        else:
            if is_left is not None and prev_node is not None:
                if is_left:
                    max_x = prev_node[0]
                else:
                    min_x = prev_node[0]

            plt.plot([min_x, max_x], [median[1], median[1]],
                     color="blue", zorder=1)
            if is_animation:
                plt.pause(1)

        if len(left_branch) != 0:
            self._plot_helper(left_branch, min_x, max_x, min_y, max_y,
                              depth + 1, median, True, is_animation)
        if len(right_branch) != 0:
            self._plot_helper(right_branch, min_x, max_x, min_y, max_y,
                              depth + 1, median, False, is_animation)

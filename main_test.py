import numpy as np
from AutonomousFleet import *

def grid(n, d_x, d_y):
    """
    Returns a grid of n points that are separated by d_x and d_y units of
    distance from each other in the x and y directions, respectively.

    This is specially useful for tests with the function leading_ships
    from AutonomousFleet.
    """
    rows = np.ceil(n**0.5)
    cols = np.round(n / rows)
    points = []
    x_coord = 0
    y_coord = 0
    for i in range(1, n + 1):
        if i % cols != 0:
            points.append([x_coord, y_coord])
            x_coord += d_x
        else:
            points.append([x_coord, y_coord])
            x_coord = 0
            y_coord += d_y
    return np.array(points)


np.random.seed(0)

# We may generate some random coordinates, or define a grid of points
coords = 2 * np.random.random_sample((80, 2)) - 1
# coords = grid(50,3,2)

# Initializing the class
T = AutonomousFleet(coords)

# Finding the 5 closest neighbors of the ship with ID 13
closest_neighbors = T.nearest_ships(13, 5, True)
print("The 5 closest neighbors of ship 13 are " +
      " ".join([str(x) for x in closest_neighbors]) + "\n")

# Finding whether there are any ships within the square of length 0.5,
# rotated -60°, and with center the ship with ID 13
ships_in_square = T.avoid_collisions(13, 0.5, -60, True)
print("The ships within the square are " +
      " ".join([str(x) for x in ships_in_square]) + "\n")

# Finding the ships with maximal or minimal x or y coordinate
north, south, east, west = T.leading_ships(True)
print("The ships with maximum x coordinate are " +
      " ".join([str(x) for x in east]))
print("The ships with maximum y coordinate are " +
      " ".join([str(x) for x in north]))
print("The ships with minimum x coordinate are " +
      " ".join([str(x) for x in west]))
print("The ships with minimum y coordinate are " +
      " ".join([str(x) for x in south]))

# Plotting the KD-Tree diagram
T.plot()

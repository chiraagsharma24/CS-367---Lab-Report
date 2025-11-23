import numpy as np
import matplotlib.pyplot as plt

# set of 10 Cities with random ordinates 
city_names = ["A","B","C","D","E","F","G","H","I","J"]

np.random.seed(42)
city_coordinates = {city: np.random.rand(2) * 100 for city in city_names}

# Function to compute the distance matrix between cities
def compute_distance_matrix(city_names):
    num_cities = len(city_names)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i, city_i in enumerate(city_names):
        for j, city_j in enumerate(city_names):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(
                    city_coordinates[city_i] - city_coordinates[city_j]
                )
    return distance_matrix

# Class to solve the Traveling Salesman Problem
class TravelingSalesmanSolver:
    def __init__(self, city_names, distance_matrix):
        self.city_names = city_names
        self.num_cities = len(city_names)
        self.distance_matrix = distance_matrix

    def calculate_tour_cost(self, tour):
        cost = sum(
            self.distance_matrix[tour[i], tour[i + 1]]
            for i in range(self.num_cities - 1)
        )
        cost += self.distance_matrix[tour[-1], tour[0]]
        return cost

    def find_optimal_tour(self, max_iterations=150000):
        best_cost = float("inf")
        best_tour = None
        current_tour = np.random.permutation(self.num_cities)

        for _ in range(max_iterations):
            i, j = np.random.choice(self.num_cities, 2, replace=False)
            current_tour[i], current_tour[j] = current_tour[j], current_tour[i]
            current_cost = self.calculate_tour_cost(current_tour)

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()

        return best_tour, best_cost

# Calculate the distance matrix
distance_matrix = compute_distance_matrix(city_names)
# Create an instance of the TSP solver
tsp_solver = TravelingSalesmanSolver(city_names, distance_matrix)
# Solve the TSP
optimal_tour, minimum_cost = tsp_solver.find_optimal_tour()
# Display the optimal tour and its cost
print("\nOptimal Tour Order:")
for idx in optimal_tour:
    print(city_names[idx], "-> ", end="")
print(city_names[optimal_tour[0]])

print("\nMinimum Path Cost:", round(minimum_cost, 3))

# Visualize the cities and the optimal tour
plt.figure(figsize=(10,8))
for city in city_names:
    plt.scatter(*city_coordinates[city], color="purple", s=100)
    plt.text(*city_coordinates[city], city, fontsize=14)

for i in range(len(optimal_tour) - 1):
    plt.plot(
        [city_coordinates[city_names[optimal_tour[i]]][0], city_coordinates[city_names[optimal_tour[i+1]]][0]],
        [city_coordinates[city_names[optimal_tour[i]]][1], city_coordinates[city_names[optimal_tour[i+1]]][1]],
        color="blue", linewidth=2
    )

plt.plot(
    [city_coordinates[city_names[optimal_tour[-1]]][0], city_coordinates[city_names[optimal_tour[0]]][0]],
    [city_coordinates[city_names[optimal_tour[-1]]][1], city_coordinates[city_names[optimal_tour[0]]][1]],
    color="blue", linewidth=2
)

plt.title("TSP - Optimal Path ", fontsize=20)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

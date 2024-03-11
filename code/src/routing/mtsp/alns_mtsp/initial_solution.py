from routing.mtsp.alns_mtsp import mtsp_helper_functions


def compute_initial_solution(current, random_state, v=2):
    num_customers = current.nb_customers
    num_vehicles = current.nr_vehicles

    # Calculate the number of required splits
    eta = num_customers // num_vehicles

    # Initialize the solution with all nodes
    solution = list(range(num_customers + 1))

    # Find the depot positions for num_vehicles-1 salesmen
    depot_indices = [(i - 1) * eta + v + 2 for i in range(1, num_vehicles)]

    # Insert depots at their corresponding indices
    for idx in reversed(depot_indices):
        solution.insert(idx, 0)

    # Add a depot to the end of the solution
    solution.append(0)

    routes = []
    route = []

    # Iterate through the list
    for node in solution:
        if node == 0:
            # Ignore 0 values as separators
            if route:
                routes.append(route)
                route = []
        else:
            # Append non-zero values to the temporary list
            route.append(node)

    current.routes = routes
    current.graph = mtsp_helper_functions.NeighborGraph(current.nb_customers)

    return current
from routing.cvrp.alns_cvrp import cvrp_helper_functions
import random


# --- init solution ---
def compute_initial_solution(current, random_state):
    routes = []
    route = []
    unvisited_customers = [i for i in range(1, current.nb_customers + 1)]
    while len(unvisited_customers) != 0:
        if len(route) == 0:
            random_customer = random.choice(unvisited_customers)
            route.append(random_customer)
            unvisited_customers.remove(random_customer)
        else:
            route_load = cvrp_helper_functions.compute_route_load(route, current.demands_data)
            unvisited_eligible_customers = cvrp_helper_functions.get_customers_that_can_be_added_to_route(route_load, current.truck_capacity,
                                                                                   unvisited_customers, current.demands_data)
            if len(unvisited_eligible_customers) == 0:
                routes.append(route)
                route = []  # new_route
                random_customer = random.choice(unvisited_customers)
                route.append(random_customer)
                unvisited_customers.remove(random_customer)
            else:
                closest_unvisited_customer = cvrp_helper_functions.get_closest_customer_to_add(route, unvisited_eligible_customers,
                                                                         current.dist_matrix_data, current.dist_depot_data)
                route.append(closest_unvisited_customer)
                unvisited_customers.remove(closest_unvisited_customer)

    if route != []:
        routes.append(route)

    current.routes = routes
    current.graph = cvrp_helper_functions.NeighborGraph(current.nb_customers)

    return current
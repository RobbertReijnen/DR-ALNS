import copy

# ----- Initial solution constructor ----------------------------------------------------------------------------------
def empty_route(state, init_node):
    state = copy.deepcopy(state)
    state.route = [init_node, init_node]

    return state
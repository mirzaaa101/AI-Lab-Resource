# Define initial state
state = [2, 1, 5, 0, 8, 4, 10, 0, 20, 10]


#  test_state = [2, 8, 1, 3, 6, 7, 5, 4]

# Define cost function
def calc_cost(my_list):
    cost = 0
    for i in range(len(my_list)):
        for j in range(i + 1, len(my_list)):
            if my_list[i] > my_list[j]:
                cost += 1
    return cost


# Define state generation function
def state_generation(current_state):
    best_state = current_state.copy()
    best_cost = calc_cost(current_state)
    for i in range(len(current_state)):
        for j in range(i + 1, len(current_state)):
            new_state1 = current_state.copy()
            new_state1[i], new_state1[j] = new_state1[j], new_state1[i]  # swapping with the forward elements of the list
            new_cost1 = calc_cost(new_state1)
            if new_cost1 < best_cost:
                best_state = new_state1
                best_cost = new_cost1
    return best_state, best_cost


def goal_test(current_state):
    return calc_cost(current_state) == 0


# Run hill climbing algorithm
while not goal_test(state):
    new_state, new_cost = state_generation(state)
    if new_cost >= calc_cost(state):
        break  # if state_generation is not generating good updated_state then it terminates
    state = new_state.copy()  # if state_generation is generating good updated_state then continue updating

print(state)

"""NOTE:comparison operator in the if statement is >= instead of > because of
as it hill climbing, so I'm not giving chance to equal valued successor otherwise it can stack within a loop"""

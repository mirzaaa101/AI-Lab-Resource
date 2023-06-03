import heapq

g_dict = {0: "S", 1: "A", 2: "B", 3: "C", 4: "D"}  # rename the nodes with number

g_list = [[0, 1, 4, 0, 0],
          [0, 0, 2, 5, 12],
          [0, 0, 0, 2, 0],
          [0, 0, 0, 0, 3],
          [0, 0, 0, 0, 0]]  # keep track of adjacent nodes

g_heuristic = [7, 6, 2, 1, 0]


class Node:
    def __init__(self, node_no, prev_node, actual_cost, total_cost):
        self.node_no = node_no
        self.prev_node = prev_node
        self.actual_cost = actual_cost
        self.total_cost = total_cost  # f(n) = g(n) + h(n)

    def __lt__(self, other):
        return self.total_cost < other.total_cost  # set priority for min priority queue


S = Node(0, None, 0, 0 + g_heuristic[0])  # create an object of starting node of S
min_queue = []  # create a list to keep the node object
heapq.heappush(min_queue, S)  # insert the starting node into min_priority queue

path = []  # list to find the path
path_cost = []  # list to find the path cost


def find_path(goal_node):
    path.append(goal_node.node_no)
    path_cost.append(goal_node.actual_cost)
    current_node = goal_node
    while current_node.prev_node is not None:
        current_node = current_node.prev_node
        path.append(current_node.node_no)
        path_cost.append(current_node.actual_cost)
    path.reverse()


while min_queue:
    N = heapq.heappop(min_queue)  # extract the min total_cost node
    if g_heuristic[N.node_no] == 0:  # heuristic value of  goal node will be zero
        find_path(N)
        break
    else:
        for i, cost in enumerate(g_list[N.node_no]):  # i for index, cost for actual_cost
            if cost != 0:  # cost not zero means there is connection with the node
                node_no1 = i
                actual_cost1 = cost
                prev_node1 = N
                total_cost1 = N.actual_cost + actual_cost1 + g_heuristic[node_no1]  # total cost from parent to child
                adj_obj = Node(node_no1, prev_node1, actual_cost1, total_cost1)  # create object of adjacent node
                heapq.heappush(min_queue, adj_obj)

print("Path", end=': ')
for i in path:
    print(g_dict[i], end='->')

print(f"\nPath Cost = {sum(path_cost)}")

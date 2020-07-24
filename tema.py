import random
import time
from copy import deepcopy

NO_RIGHT 		= 1
NO_LEFT  		= 2
DEFAULT_COST 	= 1000
REDUCE_COST		= 3
MAX_PROFIT 		= 0

north, south, east, west 	= 1, 2, 3, 4
pickup, dropoff 			= -1, -2
MAP, POS, FUEL, RIDE 		= 0, 1, 2, 3
PROFIT, CLIENTS 			= 4, 5
PICKUP, DROPOFF, BUDGET		= 0, 1, 2
x, y 						= 1, 0


height 		= 0
width 		= 0


def init_state(file):
	f = open(file)
	[height_str, width_str, capacity_str] = f.readline().split()
	
	global height, width
	height = int(height_str)
	width = int(width_str)
	capacity = int(capacity_str)
	
	[cPy_str, cPx_str] = f.readline().split()
	[cPx, cPy] = [int(cPx_str), int(cPy_str)]
	clients_no = int(f.readline().split()[0])
	clients = []

	for i in range(clients_no):
		[ssX, ssY, sdX, sdY, sb] = f.readline().split()
		[sX, sY, dX, dY, b] = [int(ssX), int(ssY), int(sdX), int(sdY), int(sb)]
		clients.append(([sX, sY], [dX, dY], b))

	f.readline()

	state = ([[0 for row in range(height)] for col in range(width)], [cPx, cPy], capacity, None, 0, clients)

	for row in range(height):
		line = f.readline().split()
		for it in range(len(line)):
			if it == 0:
				state[MAP][row][it] += NO_LEFT
			elif it == width:
				state[MAP][row][it-1] += NO_RIGHT 
			elif line[it] == '|':
				state[MAP][row][it-1] += NO_RIGHT;
				state[MAP][row][it] += NO_LEFT;

	f.readline()
	f.close()

	return state

def print_state(state):
	for row in range(height):
		l = map(lambda col: state[MAP][row][col], range(width)) 
		print("".join(str(l)))

def get_available_actions(state):
	ans = [north, south, west, east]
	clients = state[CLIENTS]

	cPx = state[POS][x]
	cPy = state[POS][y]
	
	if state[FUEL] == 0:
		return None
	if cPy == 0:
		ans.remove(north)
	if cPy == height-1:
		ans.remove(south)
	if state[MAP][cPy][cPx] == NO_RIGHT:
		ans.remove(east)
	if state[MAP][cPy][cPx] == NO_LEFT:
		ans.remove(west)
	if state[MAP][cPy][cPx] == NO_LEFT + NO_RIGHT:
		ans.remove(east)
		ans.remove(west)
	
	if state[RIDE] == None:
		for clnt in clients:
			if state[POS] == clnt[PICKUP]:
				ans.append(pickup)

	if (state[RIDE] != None) and (state[POS] == state[RIDE][DROPOFF]):
		ans.append(dropoff)

	random.shuffle(ans)
	return ans

def print_actions(lista):
	ans = []
	for elm in lista:
		if elm == south:
			ans.append("south")
		elif elm == north:
			ans.append("north")
		elif elm == east:
			ans.append("east")
		elif elm == west:
			ans.append("west")
		elif elm == pickup:
			ans.append("pickup")
		elif elm == dropoff:
			ans.append("dropoff")
	print ans

def apply_action(action, state):
	clients = state[CLIENTS]
	new_pos = deepcopy(state[POS])
	new_fuel = state[FUEL]
	
	if action > 0:
		new_fuel -= 1
		if action == south:
			new_pos[y] += 1
		if action == north:
			new_pos[y] -= 1
		if action == east:
			new_pos[x] += 1
		if action == west:
			new_pos[x] -= 1

		state = (state[MAP], new_pos, new_fuel, state[RIDE], state[PROFIT], clients)

	if action == dropoff:
		if state[RIDE] and (state[RIDE][DROPOFF] == state[POS]):
			state = drop_off(state)

	if action == pickup:
		for clnt in clients:
			if clnt[PICKUP] == state[POS]:
				aux = pick_up(state, clnt)
				if aux:
					state = aux

	return state
	

def g(state):
	clients = state[CLIENTS]
	ans = DEFAULT_COST

	# drop off location
	if state[RIDE] and state[RIDE][DROPOFF] == state[POS]:
		ans -= 150*REDUCE_COST

	# has ride, important to finish it
	if state[RIDE]:
		ans -= 75*REDUCE_COST

	# pick up location
	if state[RIDE] is None:
		for clnt in clients:
			if clnt[PICKUP] == state[POS]:
				ans -= 30*REDUCE_COST
				break

	# clients in reach
	for clnt in clients:
		if can_reach_destination(state, clnt):
			ans -= 0.1*clnt[BUDGET] * REDUCE_COST

	# profit made until now
	ans -= 10*state[PROFIT] * REDUCE_COST
	
	# ways to move around
	available_actions = get_available_actions(state)
	if available_actions:
		ans -= 2*len(available_actions) * REDUCE_COST		

	return ans

def h(state):
	ans = 0
	
	if is_final(state):
		return 0
	
	available_actions = get_available_actions(state)

	if available_actions:
		for act in available_actions:
			c = deepcopy(state)
			next_state = apply_action(act, c)
			clients = next_state[CLIENTS]
			
			# ride finished
			if act == dropoff:
				return h(next_state)-20

			# ride starts
			if act == pickup:
				return h(next_state)-30

			next_min_dist = width*height
			min_dist = width*height
			
			if next_state[RIDE] == None:

				if clients:
					for clnt in clients:
						if min_distance(next_state, clnt) < next_min_dist:
							next_min_dist = min_distance(next_state, clnt)

						if min_distance(state, clnt) < min_dist:
							min_dist = min_distance(state, clnt)
				# moves closer to the next client			
					if next_min_dist < min_dist:
						return 3+h(next_state)

			# closer to finishing the ride
			# it's better to have the least transitions possible while in ride
			if next_state[RIDE] and state[RIDE]:
				n_ride = abs(next_state[POS][x]- state[RIDE][DROPOFF][x]) + abs(next_state[POS][y] - state[RIDE][DROPOFF][y])
				c_ride = abs(state[POS][x]- state[RIDE][DROPOFF][x]) + abs(state[POS][y] - state[RIDE][DROPOFF][y])
				if n_ride < c_ride:
					# print "closer to drop"
					return 5+h(next_state)
	return DEFAULT_COST

def simple_h(state):
	if is_final(state):
		return 0

	available_actions = get_available_actions(state)
   	if available_actions == None:
   		return DEFAULT_COST
   	
   	if dropoff in available_actions:
   		c = deepcopy(state)
		next_state = apply_action(dropoff, c)
		return simple_h(next_state)

	elif pickup in available_actions:
		c = deepcopy(state)
		next_state = apply_action(pickup, c)
		return simple_h(next_state)

	elif north in available_actions:
		c = deepcopy(state)
		next_state = apply_action(north, c)
		return 10+simple_h(next_state)

	elif east in available_actions:
		c = deepcopy(state)
		next_state = apply_action(east, c)
		return 20+simple_h(next_state)

	elif south in available_actions:
		c = deepcopy(state)
		next_state = apply_action(south, c)
		return 80+simple_h(next_state)

	elif west in available_actions:
		c = deepcopy(state)
		next_state = apply_action(west, c)
		return 90+simple_h(next_state)

def min_distance(state, clnt):
	# min distance to client
	min_distance = abs(state[POS][x] - clnt[PICKUP][x]) + abs(state[POS][y] - clnt[PICKUP][y]) 
	# min distance of ride
	min_ride = abs(clnt[PICKUP][x]- clnt[DROPOFF][x]) + abs(clnt[PICKUP][y] - clnt[DROPOFF][y])

	return (min_ride+min_distance)

def can_reach_destination(state, clnt):
	# ride can be finished
	if state[FUEL] >= min_distance(state, clnt):
		return True

	return False

def is_final(state):
	clients = state[CLIENTS]
	if state[PROFIT] > 0:
		if get_available_actions(state) == None:
			return	True
		if state[FUEL] == 0:
			return True
		if clients == None and state[RIDE] == None:
			return True
		if state[RIDE]:
			return False
		for clnt in clients:
			return not can_reach_destination(state, clnt)
	return False

def pick_up(state, client):
	if state[RIDE] == None and client in state[CLIENTS] and state[POS] == client[PICKUP]:
		clients = state[CLIENTS]
		clients.remove(client)
		return (state[MAP], state[POS], state[FUEL], client, state[PROFIT], clients)
	return False

def drop_off(state):
	global MAX_PROFIT
	if state[POS] == state[RIDE][DROPOFF]:
		revenue = state[RIDE][BUDGET]
		if MAX_PROFIT < (revenue + state[PROFIT]):
			MAX_PROFIT = revenue + state[PROFIT]
		
		return (state[MAP], state[POS], state[FUEL], None, (state[PROFIT] + revenue), state[CLIENTS])
	return False

def UniformCostSearch(state):
	O = [(state, 0)]
	
	visited	= []
	parent = []
	parent.append((state, None, None))

	it = 0
	while O:
		it += 1
	
		aux = O.pop(0)
		current = aux[0]
		cur_cost = aux[1]
	
		if is_final(current):
			return (current, parent, it)

		visited.append(current)
		aux_list = O
		available_actions = get_available_actions(current)

		if available_actions:
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)
				if next_state in visited:
					continue
				parent.append((next_state, current, act))
				
				cost = g(next_state)
				aux_list.append((next_state, cur_cost+cost))
			
			sorted_list = sorted(aux_list, key=lambda x: x[1])
			O = sorted_list

	return [False, None, 0]

def BFS(state):
	O = [state]

	visited = []
	parent = []
	parent.append((state, None, None))

	it = 0
	while O:
		it += 1

		current = O.pop(0)

		if is_final(current):
			return (current, parent, it)

		# # you have to wrong to have it finished
		# if current in visited:
		# 	continue

		visited.append(current)
		available_actions = get_available_actions(current)

		if available_actions:
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)
				
				if next_state in visited:
					continue
				
				parent.append((next_state, current, act))
				O.append(next_state)

	return [False, None, 0]

def DFS(state):
	O = [state]

	visited = []
	parent = []
	parent.insert(0, (state, None, None))

	it = 0
	while O:
		it += 1

		current = O.pop(0)

		if is_final(current):
			return (current, parent, it)

		visited.append(current)
		available_actions = get_available_actions(current)

		if available_actions:
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)
		
				if next_state in visited:
					continue

				parent.append((next_state, current, act))
				O.insert(0, next_state)


	return [False, None, 0]

def DepthLimitedSearch(state, k):
	O = [(state, 0)]

	visited = []
	parent = []
	parent.append((state, None, None))

	it = 0
	while O:
		it += 1

		aux = O.pop(0)
		current = aux[0]
		depth = aux[1]+1

		if is_final(current):
			return (current, parent, it)

		visited.append(current)
		if depth >= k:
			continue

		available_actions = get_available_actions(current)
		if available_actions:
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)
				
				if next_state in visited:
					continue

				parent.append((next_state, current, act))
				O.insert(0, (next_state, depth))

	return [False, None, 0]

def IterativeDeepeningSearch(state):
	k = 0
	while 1:
		res = DepthLimitedSearch(state, k)
		if res[0] != False:
			return res
		k+=1

def GreedyBestFirstSearch(state, h):
	O = [(state, 0)]

	visited = []
	parent = []
	parent.append((state, None, None))

	it = 0
	while O:
		it += 1
	
		aux = O.pop(0)
		current = aux[0]
	
		if is_final(current):
			return (current, parent, it)

		visited.append(current)
		aux_list = O
		available_actions = get_available_actions(current)

		if available_actions:
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)

				if next_state in visited:
					continue

				parent.append((next_state, current, act))
				cost = h(next_state)

				aux_list.append((next_state, cost))

			sorted_list = sorted(aux_list, key=lambda x: x[1])
			O = sorted_list

	return [False, None, 0]

def AStar(state, h):
	O = [(state, 0, 0)]
	C = []
	
	visited = []
	parent = []
	parent.append((state, None, None))

	it = 0
	while O:
		it += 1
	
		aux = O.pop(0)
		current = aux[0]
		cur_cost = aux[1]
		cur_g = aux[2]

		if is_final(current):
			return (current, parent, it)
	
		C.append(current)
		visited.append(current)

		aux_list = O
		available_actions = get_available_actions(current)
		if available_actions:
	
			for act in available_actions:
				c = deepcopy(current)
				next_state = apply_action(act, c)

				if next_state in visited:
					continue

				g_ = g(next_state) + cur_g
				cost = g_ + h(next_state)

				cont = 0
				U = C+O
				for n_state in U:
					if n_state == next_state:
						if (g(n_state)+cur_g) < g_:
							cont = 1
						else:
							O.remove(n_state)
							C.remove(n_state)
				if cont == 1:
					continue

				parent.append((next_state, current, act))
				
				aux_list.append((next_state, cost, g_))
			
			sorted_list = sorted(aux_list, key=lambda x: x[1])
			O = sorted_list

	return [False, None, 0]

def HillClimbingSearch(state, h):
	visited = []
	parent = []
	parent.append((state, None, None))
	
	g_ = g(state)
	current_cost = g_ + h(state)
	current = (state, g_)


	max_cost = 0
	done = False
	it = 0
	while not done:
		it += 1
	
		n_state = None
		n_act = None
		n_g = 0
		
		max_cost = current_cost
		visited.append(current[0])
		
		available_actions = get_available_actions(current[0])

		if available_actions:
			for act in available_actions:
				c = deepcopy(current[0])
				next_state = apply_action(act, c)

				g_ = current[1] + g(next_state)
				cost = g_ + h(next_state)

				if cost > max_cost:
					max_cost = cost
					n_state = next_state
					n_act = act
					n_g = g_

			if n_state == None:
				done = True
			else:
				parent.append((n_state, current[0], n_act))
				current = (n_state, n_g)
				current_cost = max_cost
		else:
			break

	return (current[0], parent, it)

def build_solution(parents_list, final_state):
	solution = []

	if parents_list:
		for prt in reversed(parents_list):
			if prt[0] == final_state:
				solution.insert(0, prt[2])
				final_state = prt[1]

	return solution

def apply_algorithm(name, state, h = None):
	res = [False, None, 0]

	start_time = time.time()
	
			
	if name == "Breadth-first search":
		res = BFS(state)
	elif name == "Uniform cost search":
		res = UniformCostSearch(state)
	elif name == "Depth-first searh":
		res = DFS(state)
	elif name == "Depth-limited searh":
		res = DepthLimitedSearch(state, 20)
	elif name == "Iterative deepening search":
		res = IterativeDeepeningSearch(state)
	elif name == "Greedy best-first search":
		res = GreedyBestFirstSearch(state, h)
	elif name == "A* search":
		res = AStar(state, h)
	elif name == "Hill-climbing search":
		res = HillClimbingSearch(state, h)

	elapsed_time = time.time() - start_time

	final_state = res[0]
	cost = final_state[PROFIT]+final_state[FUEL]
	parents_list = res[1]
	states_no = res[2]

	solution = build_solution(parents_list, final_state)

	return [name, elapsed_time, states_no, cost, solution]

def run_test(test):

	blind_algorithms = ["Breadth-first search",
						"Uniform cost search",
						"Depth-first searh",
						"Depth-limited searh",
						"Iterative deepening search"]

	informat_algorithms = ["Greedy best-first search",
							"A* search",
							"Hill-climbing search"]

	print "apply blind algorithms" 

	for alg in blind_algorithms:
		state = init_state(test)

		# # print initial state and clients		
		# clients = state[CLIENTS]
		# print_state(state)

		# for clnt in clients:
		# 	print clnt

		if alg == blind_algorithms[0]:
			continue

		ans = apply_algorithm(alg, state)

		print ""
		for elm in ans:
			if type(elm) == list:
				print_actions(elm)
			else:
				print elm
	
	print "\napply informat algorithms with simple_h" 

	for alg in informat_algorithms:
		state = init_state(test)

		ans = apply_algorithm(alg, state, simple_h)

		print ""
		for elm in ans:
			if type(elm) == list:
				print_actions(elm)
			else:
				print elm


	print "\napply informat algorithms with h" 

	for alg in informat_algorithms:
		state = init_state(test)

		ans = apply_algorithm(alg, state, h)

		print ""
		for elm in ans:
			if type(elm) == list:
				print_actions(elm)
			else:
				print elm

					

if __name__ == '__main__':
	tests = ["test1.in",
			"test2.in",
			"test3.in",
			"test4.in",
			"test5.in",
			"test6.in"]
	
	for test in tests:
		print("for input: " + test)
		run_test(test)
		print ""

import copy
import time
import heapq
from ast import literal_eval
from queue import Queue

class ConnectFour:
    def __init__(self, state, depth=0, coordinates = None, columns = None):
        self.game_board = state
        self.depth = depth
        self.columns = dict()
        self.coordinates = coordinates

        if columns is None:
            self.columns = {i: 5 for i in range(7)}
        else:
            self.columns = columns

    def __lt__(self, other):  # Defines min-heap order
        return self.depth < other.depth

    def __deepcopy__(self, memo):
        new_copy = ConnectFour(copy.deepcopy(self.game_board, memo), self.depth)
        new_copy.columns = copy.deepcopy(self.columns, memo)
        return new_copy

def load_columns_state(game_board):

    # Update Columns based on Load State
    j = 0
    modified = False
    cols_state = dict()

    for i in range(7):
        count = -1
        for j in range(6):
            if game_board[j][i] == ' ':
                count += 1
            else:
                modified = True
        cols_state[i] = count

    return cols_state, modified

def restore_game_state(load_file, results):
    if load_file:
        load_file.seek(0)
        load_file.write(str([[' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ']]))
        load_file.truncate()
        load_file.flush()

def erase_results(results):
    if results:
        results.seek(0)
        results.truncate(0)
        results.flush()

def print_board(board):
    for row in board:
        print("| " + " | ".join(str(element) for element in row) + " |")
    print("\n")

def print_results(path_list, depth, count, final_time):

    print(f"Time: {final_time:.4f} seconds")

    if depth:
        print(f"Depth of Search: {depth}")
    if count:
        print(f"Nodes Explored: {count}")
    if path_list:
        print("Length of Path: ", len(path_list))
        print("Path: " + " --> ".join(map(str, path_list)) + "\n")
    else:
        print("No winning sequence found")

def save_results(file, path_list, depth, count, final_time):

    if file:

        file.write(f"Time: {final_time:.4f} seconds\n")

        if depth:
            file.write(f"Depth of Search: {depth}\n")
        if count:
            file.write(f"Nodes Explored: {count}\n")
        if path_list:
            file.write(f"Length of Path: {len(path_list)}\n")
            file.write("Path: " + " --> ".join(map(str, path_list)) + "\n\n")

        file.flush()

def is_sublist(main_list, sub_list):
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return True

    return  False

def check_goal_state(state, symbol, coordinates):

    pattern = [symbol, symbol, symbol, symbol]

    # Horizontal Matching
    row, col = coordinates
    left = max(0, col - 3)
    right = min(6, col + 3)
    horizontal = state[row][left:right + 1]
    if is_sublist(horizontal, pattern):
        return True

    # Vertical Matching
    top = max(0, row - 3)
    bottom = min(5, row + 3)
    vertical = [state[r][col] for r in range(top, bottom + 1)]
    if is_sublist(vertical, pattern):
        return True

    # Diagonal Matching
    count = 0
    diagonal = []

    while row - 1 >= 0 and col - 1 >= 0 and count <= 3:
        diagonal.append(state[row - 1][col - 1])
        row -= 1
        col -= 1
        count += 1

    count = 0
    diagonal.reverse()
    row, col = coordinates

    while row <= 5 and col <= 6 and count <= 3:
        diagonal.append(state[row][col])
        row += 1
        col += 1
        count += 1

    if is_sublist(diagonal, pattern):
        return True

    # Anti-Diagonal Matching
    count = 0
    anti_diagonal = []
    row, col = coordinates

    while row <= 5 and col >= 0 and count <= 3:
        anti_diagonal.append(state[row][col])
        row += 1
        col -= 1
        count += 1

    count = 0
    anti_diagonal.reverse()
    row, col = coordinates

    while row - 1 >= 0 and col + 1 <= 6 and count <= 3:
        anti_diagonal.append(state[row - 1][col + 1])
        row -= 1
        col += 1
        count += 1

    if is_sublist(anti_diagonal, pattern):
        return True

    return False

def check_valid_move(col, game_board):
    if col > 6 or col < 0:
        return  False
    if game_board.columns[col] == -1:
        return False

    return True

def make_move(symbol, col, state):
    state.game_board[state.columns[col]][col] = symbol
    state.columns[col] -= 1 # subtract after adding the disc
    return state.columns[col] + 1, col # returning after adding due to above subtraction

def is_empty(row, col, game_board):
    if game_board[row][col] == ' ':
        return True
    return False

def is_grid_filled(game_board):
    if any(' ' in row for row in game_board):
        return False
    return True

def find_all_points(game_board, symbol, targets):
    for i in range(6):
        for j in range(7):
            if game_board[i][j] == symbol:
                targets.add((i,j))

def find_solution_bfs(root, symbol, coordinates):

    bfs_queue = Queue(maxsize=0)
    visited = set()
    parents = dict()
    count = 0

    def generate_states_bfs(state, symbol, coordinates):

        # Move Up & Left
        if coordinates[0] - 1 >= 0 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up
        if coordinates[0] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1], state.game_board):
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1]] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1])
            duplicate.columns[coordinates[1]] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up & Right
        if coordinates[0] - 1 >= 0 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] - 1, coordinates[1] + 1, state.game_board) and coordinates[0] - state.columns[coordinates[1] + 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Left
        if coordinates[1] - 1 >= 0 and is_empty(coordinates[0], coordinates[1] - 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] - 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Right
        if coordinates[1] + 1 <= 6 and is_empty(coordinates[0], coordinates[1] + 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] + 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Left
        if coordinates[0] + 1 <= 5 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] + 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Right
        if coordinates[0] + 1 <= 5 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] + 1, coordinates[1] + 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] + 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                bfs_queue.put(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        return

    targets = set()
    targets.add(coordinates)    # Add latest move first
    find_all_points(root.game_board, symbol, targets)
    flag = False
    goal = None

    if not flag:
        for target in targets:

            root.coordinates = target
            parents[root.coordinates] = None  # Parents array for backtracking

            # Initialize queue with the initial state
            bfs_queue.put(root)
            head = None

            while not bfs_queue.empty():

                count += 1
                head = bfs_queue.get()
                visited.add(head.coordinates)
                generate_states_bfs(head, symbol, head.coordinates)

                if check_goal_state(head.game_board, symbol, head.coordinates):
                    flag = True
                    break

            if flag:
                goal = head
                break

            # Clearing queue, visited, parents
            visited.clear()
            parents.clear()
            while not bfs_queue.empty():
                bfs_queue.get()

    if not flag:
        return None, None, None

    # Finding Final Path
    current = goal.coordinates
    path = []

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path, goal.depth, count

def find_solution_dfs(root, symbol, coordinates):

    stack = []
    visited = set()
    parents = dict()
    count = 0

    def generate_states_dfs(state, symbol, coordinates):

        # Move Up & Left
        if coordinates[0] - 1 >= 0 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up
        if coordinates[0] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1], state.game_board):
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1]] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1])
            duplicate.columns[coordinates[1]] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up & Right
        if coordinates[0] - 1 >= 0 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] - 1, coordinates[1] + 1, state.game_board) and coordinates[0] - state.columns[coordinates[1] + 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Left
        if coordinates[1] - 1 >= 0 and is_empty(coordinates[0], coordinates[1] - 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] - 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Right
        if coordinates[1] + 1 <= 6 and is_empty(coordinates[0], coordinates[1] + 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] + 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Left
        if coordinates[0] + 1 <= 5 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] + 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Right
        if coordinates[0] + 1 <= 5 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] + 1, coordinates[1] + 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] + 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                stack.append(duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        return

    targets = set()
    targets.add(coordinates)    # Add latest move first
    find_all_points(root.game_board, symbol, targets)
    flag = False
    goal = None

    if not flag:
        for target in targets:

            root.coordinates = target
            parents[root.coordinates] = None  # Parents array for backtracking

            # Initialize queue with the initial state
            stack.append(root)
            head = None

            while not len(stack) == 0:

                count += 1
                head = stack.pop()
                visited.add(head.coordinates)
                generate_states_dfs(head, symbol, head.coordinates)

                if check_goal_state(head.game_board, symbol, head.coordinates):
                    flag = True
                    break

            if flag:
                goal = head
                break

            # Clearing stack, visited, parents
            visited.clear()
            parents.clear()
            stack.clear()

    if not flag:
        return None, None, None

    # Finding Final Path
    current = goal.coordinates
    path = []

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path, goal.depth, count

def find_solution_ucs(root, symbol, coordinates):

    min_heap = []
    visited = set()
    parents = dict()
    count = 0

    def generate_states_ucs(state, symbol, coordinates):

        # Move Up & Left
        if coordinates[0] - 1 >= 0 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up
        if coordinates[0] - 1 >= 0 and is_empty(coordinates[0] - 1, coordinates[1], state.game_board):
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1]] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1])
            duplicate.columns[coordinates[1]] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Up & Right
        if coordinates[0] - 1 >= 0 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] - 1, coordinates[1] + 1, state.game_board) and coordinates[0] - state.columns[coordinates[1] + 1] == 1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] - 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] - 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Left
        if coordinates[1] - 1 >= 0 and is_empty(coordinates[0], coordinates[1] - 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] - 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Right
        if coordinates[1] + 1 <= 6 and is_empty(coordinates[0], coordinates[1] + 1, state.game_board) and coordinates[
            0] - state.columns[coordinates[1] + 1] == 0:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0]][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0], coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Left
        if coordinates[0] + 1 <= 5 and coordinates[1] - 1 >= 0 and is_empty(coordinates[0] + 1, coordinates[1] - 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] - 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] - 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] - 1)
            duplicate.columns[coordinates[1] - 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        # Move Down & Right
        if coordinates[0] + 1 <= 5 and coordinates[1] + 1 <= 6 and is_empty(coordinates[0] + 1, coordinates[1] + 1,
                                                                            state.game_board) and coordinates[0] - \
                state.columns[coordinates[1] + 1] == -1:
            duplicate = copy.deepcopy(state)
            duplicate.game_board[coordinates[0] + 1][coordinates[1] + 1] = symbol
            duplicate.coordinates = (coordinates[0] + 1, coordinates[1] + 1)
            duplicate.columns[coordinates[1] + 1] -= 1
            duplicate.depth += 1

            if duplicate.coordinates not in visited:
                heapq.heappush(min_heap, duplicate)
                visited.add(duplicate.coordinates)
                parents[duplicate.coordinates] = coordinates

        return

    targets = set()
    targets.add(coordinates)    # Add latest move first
    find_all_points(root.game_board, symbol, targets)
    flag = False
    goal = None

    if not flag:
        for target in targets:

            root.coordinates = target
            parents[root.coordinates] = None  # Parents array for backtracking

            # Initialize min-heap with the initial state
            heapq.heappush(min_heap, root)
            head = None

            while not len(min_heap) == 0:

                count += 1
                head = heapq.heappop(min_heap)
                visited.add(head.coordinates)
                generate_states_ucs(head, symbol, head.coordinates)

                if check_goal_state(head.game_board, symbol, head.coordinates):
                    flag = True
                    break

            if flag:
                goal = head
                break

            # Clearing queue, visited, parents
            visited.clear()
            parents.clear()
            min_heap.clear()

    if not flag:
        return None, None, None

    # Finding Final Path
    current = goal.coordinates
    path = []

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path, goal.depth, count

def start_game(root, results, load_file):

    print("\n************************************************ Welcome to ConnectFour! ************************************************\n")
    mode = int(input("\nPlease Choose The Desired Space Search Algorithm --> 1 = BFS, 2 = DFS, 3 = UCS: "))
    print("\n")
    print_board(root.game_board)

    while not is_grid_filled(root.game_board):

        # Player 1

        while True:
            player1 = int(input("Please enter the column to add the disc: "))
            if check_valid_move(player1, root):
                break
            else:
                print("Please provide a valid column")

        coordinates1 = make_move('X', player1, root)
        print("Move: ", coordinates1)

        if check_goal_state(root.game_board, 'X', coordinates1):
            print("Player1 Wins!\n")
            print_board(root.game_board)
            restore_game_state(load_file, results)   # restoring game board after win
            break

        start_time = time.time()

        if mode == 1:
            print("\n******* Breadth First Search *******\n")
            paths, depth, count = find_solution_bfs(root, 'X', coordinates1)
        elif mode == 2:
            print("\n******* Depth First Search *******\n")
            paths, depth, count = find_solution_dfs(root, 'X', coordinates1)
        else:
            print("\n******* Uniform Cost Search *******\n")
            paths, depth, count = find_solution_ucs(root, 'X', coordinates1)

        final_time = time.time() - start_time
        print_results(paths, depth, count, final_time)
        save_results(results, paths, depth, count, final_time)
        print_board(root.game_board)

        # Player 2

        while True:
            player2 = int(input("Please enter the column to add the disc: "))
            if check_valid_move(player2, root):
                break
            else:
                print("Please provide a valid column")

        coordinates2 = make_move('O', player2, root)
        print("Move: ", coordinates1)

        if check_goal_state(root.game_board, 'O', coordinates2):
            print("Player2 Wins!\n")
            print_board(root.game_board)
            restore_game_state(load_file, results)   # restoring game board after win
            break

        # Applying BFS
        print("\n******* Breadth First Search *******\n")
        start_time = time.time()

        if mode == 1:
            paths, depth, count = find_solution_bfs(root, 'O', coordinates2)
        elif mode == 2:
            print("\n******* Depth First Search *******\n")
            paths, depth, count = find_solution_dfs(root, 'X', coordinates1)
        else:
            print("\n******* Uniform Cost Search *******\n")
            paths, depth, count = find_solution_ucs(root, 'X', coordinates2)

        final_time = time.time() - start_time
        print_results(paths, depth, count, final_time)
        save_results(results, paths, depth, count, final_time)
        print_board(root.game_board)

        # Save Game
        save_flag = input("\nDo You Want To Save & Quit? (Y/N): ")
        if save_flag.upper() == 'Y':
            load_file.seek(0)
            load_file.write(str(root.game_board))
            load_file.truncate()
            print("Game Saved. Thanks for playing!\n")
            break

# Loading Game from Save File
load_file = open('save_file.txt', "r+")
load_state = literal_eval(load_file.read())
cols_state, modified = load_columns_state(load_state)
results = open('results.txt', "a")
if not modified:
    erase_results(results)

Game = ConnectFour(load_state, 0, None, cols_state)

start_game(Game, results, load_file)
load_file.close()
results.close()
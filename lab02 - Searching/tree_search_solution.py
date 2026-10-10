import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

# Biến cấu hình thứ tự duyệt hướng
_search_order = "NESW"
_random_order = False
heuristic = None

def set_order(order="NESW", random=False):
    global _search_order, _random_order
    _search_order = order
    _random_order = random

def get_available_directions():
    if _random_order:
        dirs = list(_search_order)
        random.shuffle(dirs)
        return dirs
    return list(_search_order)

def find_maze_marker(maze, marker):
    indices = np.where(maze == marker)
    if len(indices[0]) > 0:
        return (indices[0][0], indices[1][0])
    marker_lower = marker.lower()
    indices = np.where(maze == marker_lower)
    if len(indices[0]) > 0:
        return (indices[0][0], indices[1][0])
    return None

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.f_cost = path_cost + heuristic

    def expand(self, maze):
        children = []
        directions = {
            'N': (-1, 0),
            'S': (1, 0),
            'E': (0, 1),
            'W': (0, -1)
        }
        
        order = get_available_directions()
        for action in order:
            dr, dc = directions[action]
            next_state = (self.state[0] + dr, self.state[1] + dc)
            
            if (0 <= next_state[0] < maze.shape[0]) and (0 <= next_state[1] < maze.shape[1]):
                if maze[next_state] != 'X' and maze[next_state] != 'x':
                    cost = self.path_cost + 1
                    children.append(Node(state=next_state, parent=self, action=action, path_cost=cost))
        return children

    def get_path(self):
        node, path = self, []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def get_actions(self):
        node, actions = self, []
        while node and node.action:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]

def manhattan(node, goal):
    state = node.state if hasattr(node, 'state') else node
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def min_index(lst):
    if not lst:
        return None
    return lst.index(min(lst))

def show_maze(maze):
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])
    m_copy = np.copy(maze)
    start = find_maze_marker(m_copy, 'S')
    goal = find_maze_marker(m_copy, 'G')
    m_copy[m_copy == ' '] = 0
    m_copy[m_copy == 'X'] = 1
    m_copy[m_copy == 'x'] = 1
    m_copy[m_copy == 'S'] = 2
    m_copy[m_copy == 'G'] = 3
    m_copy = m_copy.astype(int)
    
    fig, ax = plt.subplots()
    ax.imshow(m_copy, cmap=cmap, norm=colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
    if start:
        plt.text(start[1], start[0], "S", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')
    if goal:
        plt.text(goal[1], goal[0], "G", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')
    plt.show()

def show_path(maze, result):
    if result.get('path') is not None:
        print(f"Path length: {len(result['path'])-1}")
        print(f"Reached squares: {len(result.get('reached', []))}")
        print(f"Action sequence: {result.get('actions', [])}")
        if 'max_depth' in result:
            print(f"Max depth: {result['max_depth']}")
        if 'max_frontier' in result:
            print(f"Max frontier: {result['max_frontier']}")
        
        m_copy = np.copy(maze)
        if result.get('reached') is not None:
            for state in result['reached']:
                if m_copy[state] not in ['S', 'G', 'X', 'x']:
                    m_copy[state] = '.'
        if result.get('path') is not None:
            for state in result['path']:
                if m_copy[state] not in ['S', 'G', 'X', 'x']:
                    m_copy[state] = 'P'
                    
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])
        start = find_maze_marker(m_copy, 'S')
        goal = find_maze_marker(m_copy, 'G')
        
        m_copy[m_copy == ' '] = 0
        m_copy[m_copy == 'X'] = 1
        m_copy[m_copy == 'x'] = 1
        m_copy[m_copy == 'S'] = 2
        m_copy[m_copy == 'G'] = 3
        m_copy[m_copy == 'P'] = 4
        m_copy[m_copy == '.'] = 5
        m_copy = m_copy.astype(int)
        
        fig, ax = plt.subplots()
        ax.imshow(m_copy, cmap=cmap, norm=colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
        if start:
            plt.text(start[1], start[0], "S", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')
        if goal:
            plt.text(goal[1], goal[0], "G", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')
        plt.show()
    else:
        print("No solution found.")
    return result

def capture_frame(current_maze, current_explored, current_path):
    m = np.copy(current_maze)
    for state in current_explored:
        if m[state] not in ['S', 'G', 'X', 'x']:
            m[state] = '.'
    if current_path:
        for state in current_path:
            if m[state] not in ['S', 'G', 'X', 'x']:
                m[state] = 'P'
    return m

def best_first_search(maze, strategy="BFS", W=1.0, debug=False, vis=False, animation=False):
    start = find_maze_marker(maze, 'S')
    goal = find_maze_marker(maze, 'G')
    
    if start is None or goal is None:
        start = (0, 0)
        goal = (maze.shape[0]-1, maze.shape[1]-1)

    start_node = Node(state=start)
    frontier = [start_node]
    explored = set()
    states_explored_count = 0
    maze_anim_frames = []

    # Khởi tạo thông số theo dõi
    max_frontier = len(frontier)
    max_depth = 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        if strategy == "BFS":
            node = frontier.pop(0)
        elif strategy in ["GBFS", "A*"]:
            frontier.sort(key=lambda n: (n.path_cost + W * n.heuristic) if strategy == "A*" else n.heuristic)
            node = frontier.pop(0)
        else:
            node = frontier.pop(0)

        max_depth = max(max_depth, node.path_cost)

        if node.state not in explored:
            explored.add(node.state)
            states_explored_count += 1
            if animation:
                maze_anim_frames.append(capture_frame(maze, explored, None))

        if node.state == goal:
            path_nodes = node.get_path()
            if animation:
                maze_anim_frames.append(capture_frame(maze, explored, path_nodes))
            return {
                'path': path_nodes,
                'actions': node.get_actions(),
                'reached': list(explored),
                'states_explored': states_explored_count,
                'path_length': len(node.get_actions()),
                'max_depth': max_depth,
                'max_frontier': max_frontier,
                'maze_anim': maze_anim_frames if animation else None
            }
            
        for child in node.expand(maze):
            if child.state not in explored:
                h_val = manhattan(child, goal)
                if heuristic is not None:
                    h_val = heuristic(child, goal)
                child.heuristic = h_val
                child.f_cost = child.path_cost + W * child.heuristic
                frontier.append(child)

    return {
        'path': None, 
        'actions': None, 
        'reached': list(explored), 
        'states_explored': states_explored_count, 
        'path_length': 0,
        'max_depth': max_depth,
        'max_frontier': max_frontier,
        'maze_anim': maze_anim_frames if animation else None
    }

def DFS(maze, vis=False, max_tries=50000, debug_reached=False, limit=None, frontier_option=1, check_cycle=True, animation=False):
    start = find_maze_marker(maze, 'S')
    goal = find_maze_marker(maze, 'G')
    
    if start is None or goal is None:
        start = (0, 0)
        goal = (maze.shape[0]-1, maze.shape[1]-1)

    start_node = Node(state=start)
    frontier = [start_node]
    reached = {start} if check_cycle else set()
    states_explored_count = 0
    tries = 0
    maze_anim_frames = []

    max_frontier = len(frontier)
    max_depth = 0

    while frontier and tries < max_tries:
        max_frontier = max(max_frontier, len(frontier))
        tries += 1
        node = frontier.pop()
        states_explored_count += 1
        
        max_depth = max(max_depth, node.path_cost)

        if animation:
            maze_anim_frames.append(capture_frame(maze, reached, None))

        if node.state == goal:
            path_nodes = node.get_path()
            if animation:
                maze_anim_frames.append(capture_frame(maze, reached, path_nodes))
            return {
                'path': path_nodes,
                'actions': node.get_actions(),
                'reached': list(reached),
                'states_explored': states_explored_count,
                'path_length': len(node.get_actions()),
                'max_depth': max_depth,
                'max_frontier': max_frontier,
                'maze_anim': maze_anim_frames if animation else None
            }

        children = node.expand(maze)
        children.reverse()
        
        for child in children:
            if not check_cycle:
                frontier.append(child)
            else:
                if child.state not in reached:
                    reached.add(child.state)
                    frontier.append(child)

    return {
        'path': None, 
        'actions': None, 
        'reached': list(reached), 
        'states_explored': states_explored_count, 
        'path_length': 0,
        'max_depth': max_depth,
        'max_frontier': max_frontier,
        'maze_anim': maze_anim_frames if animation else None
    }

def IDS(maze, max_tries=50000, vis=False, frontier_option=1, animation=False):
    total_explored = 0
    for depth in range(1, 50):
        result = DLS(maze, limit=depth, max_tries=max_tries)
        if result is not None:
            total_explored += result['states_explored']
            if result['path'] is not None:
                result['states_explored'] = total_explored
                if animation:
                    result['maze_anim'] = [capture_frame(maze, result['path'], result['path'])]
                return result
        else:
            break
            
    return {'path': None, 'actions': None, 'reached': [], 'states_explored': total_explored, 'path_length': 0, 'maze_anim': []}

def DLS(maze, limit, max_tries=50000):
    start = find_maze_marker(maze, 'S')
    goal = find_maze_marker(maze, 'G')
    
    if start is None or goal is None:
        start = (0, 0)
        goal = (maze.shape[0]-1, maze.shape[1]-1)

    start_node = Node(state=start, path_cost=0)
    frontier = [start_node]
    states_explored_count = 0
    tries = 0

    while frontier and tries < max_tries:
        tries += 1
        node = frontier.pop()

        if node.state == goal:
            return {
                'path': node.get_path(),
                'actions': node.get_actions(),
                'states_explored': states_explored_count,
                'path_length': len(node.get_actions())
            }

        if node.path_cost < limit:
            states_explored_count += 1
            children = node.expand(maze)
            children.reverse()
            for child in children:
                frontier.append(child)

    return {'path': None, 'actions': None, 'states_explored': states_explored_count, 'path_length': 0}

def animate_maze(result, repeat=False):
    def maze_to_matrix(maze):
        maze = np.copy(maze)
        maze[maze == ' '] = 0
        maze[maze == 'X'] = 1
        maze[maze == 'x'] = 1
        maze[maze == 'S'] = 2
        maze[maze == 'G'] = 3
        maze[maze == 'P'] = 4
        maze[maze == '.'] = 5
        maze[maze == 'F'] = 6
        return maze.astype(int)

    if result.get('path') is not None:       
        print(f"Path length: {len(result['path'])-1}")
        print(f"Reached squares: {len(result.get('reached', []))}")
        print(f"Action sequence: {result.get('actions', [])}")
        if 'max_depth' in result:
            print(f"Max depth: {result['max_depth']}")
        if 'max_frontier' in result:
            print(f"Max frontier: {result['max_frontier']}")
    else:
        print("No solution found.")
    
    mazes = result.get('maze_anim')
    if not mazes:
        return None
    
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])
    goal = find_maze_marker(mazes[0], 'G')
    start = find_maze_marker(mazes[0], 'S')
    
    fig, ax = plt.subplots()
    im = ax.imshow(maze_to_matrix(mazes[0]), cmap=cmap, norm=colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
 
    if start:
        plt.text(start[1], start[0], "S", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')
    if goal:
        plt.text(goal[1], goal[0], "G", fontsize=10, color="white", horizontalalignment='center', verticalalignment='center')

    def step(i):  
        im.set_array(maze_to_matrix(mazes[i]))
        return [im]
 
    ani = animation.FuncAnimation(fig, step, frames=len(mazes), repeat=repeat)
    plt.close()
    return ani
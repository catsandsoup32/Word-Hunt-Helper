import trie as trie
from trie import TrieNode
from typing import Optional
import copy
from character_recognition.custom_parse import get_grid
from PIL import Image

NUM_ROWS = NUM_COLS = 4
directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def build_trie(root):
    with open("grid_solver/word_list.txt", "r") as f:
        for line in f:
            line = line.strip() 
            trie.insert(root, line)


def get_valid_directions(grid, curr_pos, positions, node):
    """Returns a list of (x,y) coords that are in-bounds, non-visited, and that are viable prefixes"""
    valid_directions = []

    for coord in directions:
        nr, nc = curr_pos[0] + coord[0], curr_pos[1] + coord[1]
        if (0 <= nr < NUM_ROWS and 0 <= nc < NUM_COLS and (nr, nc) not in positions): 
            if node.children[ord(grid[nr][nc]) - ord('a')] is not None:
                valid_directions.append((nr, nc))
    return valid_directions


def search_all(grid, word_hash, curr_pos, positions, prefix, node):
    """
    Args:
        grid (2-D array): Board represented as grid
        word_hash (dict): Maps each word to a set of positions
        curr_pos (tuple): Row and column
        positions (set of tuples): Positions up to current, inclusive  
        prefix (string): Prefix including current char
        node (TrieNode): Current node
    """
    if node.end_of_word and not (prefix in word_hash): 
        word_hash[prefix] = tuple(positions)

    valid_directions = get_valid_directions(grid, curr_pos, positions, node) 
    if not valid_directions: 
        return 
    
    for row, col in valid_directions:
        positions[(row, col)] = None
        new_char = grid[row][col]
        search_all(grid, word_hash, (row, col), positions, prefix + new_char, node.children[ord(new_char) - ord('a')])
        positions.pop((row, col))


def solve(pause_print: bool, photo: Optional[Image.Image]):
    """
    Args: 
        pause_print (bool): Whether to print solutions drawn on grid.
        photo (PIL image or None): If None, grid must be entered manually. 
    """
    grid = []
    word_hash = {}

    if photo is None: 
        print("(all lowercase input)")
        for i in range(1, NUM_ROWS+1):
            n = 0
            while (n != NUM_ROWS):
                row = input(f"Enter row {i}: ") 
                n = len(row)
                grid.append(list(row))
    else:
        get_grid (
            image_path_or_obj = photo,
            show_process = False,
            grid = grid
        )
    
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            positions = {}
            positions[(row, col)] = None
            curr_char = grid[row][col]
            search_all(grid, word_hash, (row, col), positions, curr_char, root.children[ord(curr_char)-ord('a')])


    sorted_hash = sorted(word_hash.items(), key=lambda item: len(item[0]), reverse=True)
    print(f"\nGrid:\n")
    for i in range(NUM_ROWS): print("   ".join(grid[i]))
    print(f"\nFound words:\n{sorted_hash}")

    if pause_print:
        for word, positions in sorted_hash:
            temp_grid = copy.deepcopy(grid)
            for i, coord in enumerate(positions):
                r, c = coord[0], coord[1]
                curr_char = temp_grid[r][c]
                temp_grid[r][c] = "\033[31m" + curr_char + "\033[0m"
            
            print("\033[31m" + word + "\033[0m")
            for i in range(NUM_ROWS):
                print("   ".join(temp_grid[i]))
            
            input("Press enter for next")
    
    return sorted_hash

    
#  For laptop only
if __name__ == "__main__":
    root = TrieNode()
    build_trie(root)
    solve(True, Image.open("grid_solver/character_recognition/JsxLT.jpg"))
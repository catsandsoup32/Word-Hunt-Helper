# main file to solve the Word Hunt grid
# then send solution to Arduino

from trie import TrieNode
from solver import build_trie, solve
from character_recognition.custom_parse import get_grid

root = TrieNode()
build_trie(root)
grid = []

# Camera code to take a photo
# On button press?
photo = "placeholder"
# set grid to empty array


solutions = solve(grid)


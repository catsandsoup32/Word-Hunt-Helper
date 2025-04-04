class TrieNode():
    def __init__(self):
        self.children = [None] * 26 
        self.end_of_word = False

def insert(root, word):
    """Creates new branches for word paths, if not already present, and marks new word ending"""
    curr = root
    for c in word:
        idx = ord(c) - ord('a')
        if (curr.children[idx] is None): 
            new = TrieNode()
            curr.children[idx] = new
        curr = curr.children[idx] 
    curr.end_of_word = True 



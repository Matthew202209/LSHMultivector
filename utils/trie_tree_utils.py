import pickle


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False
        self.data = None # 从根节点到当前节点的所有路径

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.result = [[],[]]

    def insert(self, binary_code, data):
        node = self.root
        for bit in binary_code:
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
        node.is_leaf = True
        node.data = data

    def search(self, binary_code, max_distance):
        self.dfs(binary_code, self.root, '', 0, max_distance)
        return self.result


    def dfs(self, binary_code, node, current_code, current_distance, max_distance):
        if current_distance > max_distance:
            return
        if node.is_leaf:
            self.result[0]+= node.data[0]
            self.result[1]+= node.data[1] # 直接产出所有子集
        for bit, child in node.children.items():
            new_code = current_code + bit
            new_distance = current_distance + (bit != binary_code[len(new_code) - 1])
            if new_distance <= max_distance:
                self.dfs(binary_code, child, new_code, new_distance, max_distance)

    def reset_result(self):
        self.result = [[], []]

if __name__ == '__main__':
    trie = Trie()
    trie.insert("000", [[1, 2, 3], [12, 2, 3]])
    trie.insert("001", [[4, 5, 6], [4, 15, 6]])
    trie.insert("111", [[7, 8, 9], [7, 18, 9]])

    with open("test.pkl", "wb") as f:
        pickle.dump(trie, f)

    with open("test.pkl", "rb") as f:
        loaded_trie = pickle.load(f)
    query_code = "000"
    max_distance = 3
    result = loaded_trie.search(query_code, max_distance)
    trie.reset_result()
    print(result)
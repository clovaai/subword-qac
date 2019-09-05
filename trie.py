"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""


class Node(object):
    def __init__(self):
        self.children = {}
        self.count = 0

    def child(self, char, create=True):
        if char not in self.children:
            if not create:
                return None
            self.children[char] = Node()
        return self.children[char]


class Trie(object):
    def __init__(self, data=None):
        self.root = Node()
        self.n_nodes = 1

        if data:
            self.data = sorted(data)
            self.add_multiple(self.root, 0, 0, len(data))

    def find(self, string, create=True):
        node = self.root
        for x in string:
            node = node.child(x, create)
            if node is None:
                return None
        return node

    def add(self, string):
        node = self.find(string, True)
        node.count += 1

    def add_multiple(self, node, depth, l, r):
        children = [['', 0]]
        for i in range(l, r):
            if len(self.data[i]) == depth:
                children[0][1] += 1
            else:
                c = self.data[i][depth]
                if children[-1][0] == c:
                    children[-1][1] += 1
                else:
                    children.append([c, 1])
        node.count += children[0][1]
        s = l + children[0][1]
        for x in children[1:]:
            e = s + x[1]
            self.add_multiple(node.child(x[0]), depth + 1, s, e)
            s = e

    def traverse(self, node, prefix, n_candidates, min_freq):
        if hasattr(node, 'mpc'):
            return
        node.mpc = [(node.count, prefix)] if node.count >= min_freq else []
        for c, child in node.children.items():
            self.traverse(child, prefix + c, n_candidates, min_freq)
            node.mpc.extend(child.mpc)
        node.mpc = sorted(node.mpc, reverse=True)[:n_candidates]

    def get_mpc(self, prefix, n_candidates=10, min_freq=1):
        node = self.find(prefix, False)
        if node is None:
            return []
        self.traverse(node, prefix, n_candidates, min_freq)
        return [completion for _, completion in node.mpc]

# class TrieItem:
#     def __init__(self, ):
import itertools


class ClosedTrieNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.children = dict()
        self.item = None
        self.has_children = False

    def __len__(self):
        return self.depth

    def insert(self, list_, item=None):
        if len(list_) == len(self):
            if not self.has_children:
                self.item = item
        else:
            self.has_children = True
            self.item = None
            c = list_[self.depth]
            if c not in self.children:
                self.children[c] = ClosedTrieNode(depth=len(self) + 1)

            self.children[c].insert(list_, item)

    def contains(self, list_):
        if len(list_) == len(self):
            return True

        return self.children[list_[len(self)]].contains(list_) if list_[len(self)] in self.children else False

    def items(self):
        children_items = list(itertools.chain.from_iterable([child.items() for k, child in self.children.items()]))
        my_item = [] if self.item is None else [self.item]
        return children_items + my_item


if __name__ == '__main__':
    trie = ClosedTrieNode()
    trie.insert('123', ('123', 'a'))
    print(trie.items())
    trie.insert('1232', ('1232', 'b'))
    print(trie.items())
    trie.insert('12', ('12', 'c'))
    print(trie.items())
    trie.insert('22', ('22', 'e'))
    print(trie.items())
    trie.insert('21', ('21', 'f'))
    print(trie.items())
    trie.insert('222', ('222', 'g'))
    print(trie.items())
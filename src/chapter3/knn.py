
class BTreeNode:
    '''
    二叉树结点
    '''
    def __init__(self, left, right, index, \
            key , leftindex, rightindex):
        '''

        二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        '''
        self.leftindex = leftindex
        self.rightindex = rightindex
        self.left = left
        self.right = right
        self.index = index
        self.key = key

class BinaryTree:
    '''
    二叉树
    '''
    def __init__(self):
        '''
        二叉树
        '''
        self.lastnode = None
        self.root = None
        self.nodes = []

    def addnode(self, leftindex : int, rightindex : int, selfindex : int, selfkey):
        '''
        加入二叉树结点

        Args
        ===
        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        `selfindex` : 结点自身索引值

        `selfkey` : 结点自身键值

        '''
        leftnode = self.findnode(leftindex)
        rightnode = self.findnode(rightindex)
        x = BTreeNode(leftnode, rightnode, selfindex, \
            selfkey, leftindex, rightindex)
        self.nodes.append(x)
        self.lastnode = x
        return x
        
    def renewall(self) -> None:
        '''
        更新/连接/构造二叉树
        '''
        for node in self.nodes:
            node.left = self.findnode(node.leftindex)
            node.right = self.findnode(node.rightindex)
    
    def findleftrightnode(self, node : BTreeNode) -> list:
        '''
        找出二叉树某结点的所有子结点

        Args
        ===
        `node` : BTreeNode : 某结点
        '''
        array = []
        if node != None:
            # 递归找到左儿子所有的结点
            leftnodes = self.findleftrightnode(node.left)
            # 递归找到右兄弟所有的结点
            rightnodes = self.findleftrightnode(node.right)
            if leftnodes != None and len(leftnodes) != 0:
                # 连接两个集合
                array = array + leftnodes
            if rightnodes != None and len(rightnodes) != 0:
                # 连接两个集合
                array= array + rightnodes
            # 将自己本身的结点也加入集合
            array.append({ "index":node.index, "key" : node.key})
            if len(array) == 0:
                return None
            return array
        return None

    def all(self) -> list:
        '''
        返回二叉树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def keys(self) -> list:
        '''
        返回二叉树中所有结点键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append(node.key)
        return array

    def findnode(self, index : int):
        '''
        根据索引寻找结点`O(n)`

        Args
        ===
        `index` : 索引值
        '''
        if index == None:
            return None
        for node in self.nodes:
            if node.index == index:
                return node
        return None

class KDTreeNode:
    pass

class KDTree(BinaryTree):
    """
    kd二叉树
    """
    pass

class SKLearnKNeighbors:
    pass

def main():
    pass

if __name__ == '__main__':
    main()

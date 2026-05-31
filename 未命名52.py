from collections import deque


# ======================
# 二叉树结点定义
# ======================
class BiTNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None


# ======================
# 根据先序字符串构造二叉树（'#' 表示空节点）
# 示例: "ABD##E##C#F##" → A(B(D,E), C(∅,F))
# ======================
def create_tree(preorder):
    """
    使用迭代器递归构建二叉树。
    """
    def build(it):
        val = next(it)
        if val == '#':
            return None
        node = BiTNode(val)
        node.lchild = build(it)
        node.rchild = build(it)
        return node

    return build(iter(preorder))


# ======================
# 1. 计算单分支结点个数
# 单分支：只有左孩子 或 只有右孩子
# ======================
def count_single(bt):
    if bt is None:
        return 0
    cnt = 0
    if (bt.lchild and not bt.rchild) or (not bt.lchild and bt.rchild):
        cnt = 1
    return cnt + count_single(bt.lchild) + count_single(bt.rchild)


# ======================
# 2. 先序遍历并输出每个结点的层次
# ======================
def preorder_with_level(bt, level=1):
    if bt is None:
        return
    print(f"结点 {bt.data} 在第 {level} 层")
    preorder_with_level(bt.lchild, level + 1)
    preorder_with_level(bt.rchild, level + 1)


# ======================
# 3. 判断两棵二叉树是否等价（结构和数据都相同）
# ======================
def is_equal(t1, t2):
    if not t1 and not t2:
        return True
    if not t1 or not t2:
        return False
    if t1.data != t2.data:
        return False
    return is_equal(t1.lchild, t2.lchild) and is_equal(t1.rchild, t2.rchild)


# ======================
# 4. 交换二叉树的所有左右子树
# ======================
def swap_lr(bt):
    if bt is None:
        return
    bt.lchild, bt.rchild = bt.rchild, bt.lchild
    swap_lr(bt.lchild)
    swap_lr(bt.rchild)


# ======================
# 5. 判断是否为完全二叉树
# 完全二叉树：按层遍历时，一旦出现空节点，后续不能有非空节点
# ======================
def is_complete(bt):
    if bt is None:
        return True

    queue = deque([bt])
    flag = False  # 标记是否已遇到空节点

    while queue:
        node = queue.popleft()
        if node:
            if flag:
                # 已经遇到过空节点，现在又出现非空 → 不是完全二叉树
                return False
            queue.append(node.lchild)
            queue.append(node.rchild)
        else:
            flag = True

    return True


# ======================
# 主程序：测试所有功能
# ======================
if __name__ == '__main__':
    # 构造测试树: A(B(D, E), C(#, F))
    # 先序序列: A B D # # E # # C # F # #
    # 字符串表示: "ABD##E##C#F##"
    preorder = "ABD##E##C#F##"
    bt = create_tree(preorder)

    print("=== 1. 单分支结点个数 ===")
    print(count_single(bt))  # 应输出 1（结点 C 是单分支）

    print("\n=== 2. 先序遍历输出层次 ===")
    preorder_with_level(bt)

    print("\n=== 3. 判断两棵树是否等价 ===")
    bt2 = create_tree(preorder)
    print(is_equal(bt, bt2))  # True

    print("\n=== 4. 交换左右子树 ===")
    swap_lr(bt)
    preorder_with_level(bt)  # 交换后：A(C(F), B(E, D))

    print("\n=== 5. 是否为完全二叉树 ===")
    # 构造一棵非完全二叉树: A(B(C,D), E(∅,F))
    bt3 = create_tree("ABC##DE###F##")
    print(is_complete(bt3))  # False
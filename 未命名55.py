# 定义链表节点
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None


# ========== 主测试流程 ==========
# 手动创建链表: 1 -> 2 -> 6 -> 3 -> 4 -> 5 -> 6
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(6)
head.next.next.next = ListNode(3)
head.next.next.next.next = ListNode(4)
head.next.next.next.next.next = ListNode(5)
head.next.next.next.next.next.next = ListNode(6)

print("原链表: ", end="")
current = head
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目1】删除所有值为6的节点")
# 使用虚拟头节点法删除值为6的节点
dummy = ListNode(0)
dummy.next = head
prev = dummy
curr = head

while curr:
    if curr.val == 6:
        prev.next = curr.next  # 跳过当前节点
    else:
        prev = curr
    curr = curr.next

head = dummy.next

print("删除后: ", end="")
current = head
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目2】将负数移到其他节点之前")
# 当前链表中没有负数，所以结果不变
neg_head = ListNode(0)  # 负数链虚拟头
pos_head = ListNode(0)  # 非负数链虚拟头
neg_tail = neg_head
pos_tail = pos_head
curr = head

while curr:
    next_node = curr.next
    curr.next = None
    if curr.val < 0:
        neg_tail.next = curr
        neg_tail = curr
    else:
        pos_tail.next = curr
        pos_tail = curr
    curr = next_node

# 合并：负数在前，非负在后
neg_tail.next = pos_head.next
head = neg_head.next

print("调整后: ", end="")
current = head
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目3】删除有序链表中的重复值")
# 注意：当前链表 [1,2,3,4,5] 已无重复，但算法适用于有序去重
curr = head
while curr and curr.next:
    if curr.val == curr.next.val:
        curr.next = curr.next.next
    else:
        curr = curr.next

print("去重后: ", end="")
current = head
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目4】在第一个最大值前插入99")
# 找第一个最大值（从左到右首次出现的最大值）
max_val = head.val
max_node = head
prev_of_max = None
pred = head
curr = head.next

while curr:
    if curr.val > max_val:
        max_val = curr.val
        max_node = curr
        prev_of_max = pred
    pred = curr
    curr = curr.next

# 插入99
new_node = ListNode(99)
if prev_of_max is None:
    # 最大值是头节点
    new_node.next = head
    head = new_node
else:
    new_node.next = max_node
    prev_of_max.next = new_node

print("插入后: ", end="")
current = head
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目5】删除A中在B中出现的元素")

# 创建链表 A: 1 -> 2 -> 3 -> 4 -> 5
head_a = ListNode(1)
head_a.next = ListNode(2)
head_a.next.next = ListNode(3)
head_a.next.next.next = ListNode(4)
head_a.next.next.next.next = ListNode(5)

# 创建链表 B: 2 -> 4 -> 6 -> 8
head_b = ListNode(2)
head_b.next = ListNode(4)
head_b.next.next = ListNode(6)
head_b.next.next.next = ListNode(8)

print("链表A: ", end="")
current = head_a
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next

print("链表B: ", end="")
current = head_b
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next

# 将B的值存入集合（O(1)查找）
b_values = set()
curr = head_b
while curr:
    b_values.add(curr.val)
    curr = curr.next

# 删除A中出现在B中的节点
dummy_a = ListNode(0)
dummy_a.next = head_a
prev = dummy_a
curr = head_a

while curr:
    if curr.val in b_values:
        prev.next = curr.next
    else:
        prev = curr
    curr = curr.next

head_a = dummy_a.next

print("删除后A: ", end="")
current = head_a
while current:
    print(current.val, end=" -> " if current.next else "\n")
    current = current.next


print("\n" + "=" * 50)
print("【题目6】查找倒数第K个节点")

# 在主链表 head 中查找倒数第2个节点（此时 head 是：1->2->3->4->99->5）
k = 2
fast = slow = head

# 快指针先走 k 步
for _ in range(k):
    if not fast:
        break
    fast = fast.next

# 双指针同步移动
while fast:
    fast = fast.next
    slow = slow.next

if slow:
    print(f"倒数第 {k} 个节点的值是: {slow.val}")
else:
    print(f"倒数第 {k} 个节点不存在")


print("\n" + "=" * 50)
print("测试结束")
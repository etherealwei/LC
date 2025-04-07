from cgitb import reset
from distutils.command.check import check
from inspect import stack
from random import random
from typing import *
from unicodedata import digit
from leetcode_0_types import *


class NumArray:
    def __init__(self, nums: List[int]):
        self.N = len(nums)
        self.arr = [0] * 2 * self.N
        for i in range(self.N):
            self.arr[i + self.N] = nums[i]

        for i in range(self.N - 1, 0, -1):
            self.arr[i] = self.arr[i << 1] + self.arr[(i << 1) + 1]

    def update(self, index: int, val: int) -> None:
        i = index + self.N
        old_val = self.arr[i]
        self.arr[i] = val
        while i > 0:
            i = i >> 1
            self.arr[i] += val - old_val


    def sumRange(self, left: int, right: int) -> int:
        l, r = left + self.N, right + self.N + 1

        result = 0
        while l < r:
            if l & 1:
                result += self.arr[l]
                l += 1
            if r & 1:
                r -= 1
                result += self.arr[r]
            l = l >> 1
            r = r >> 1
        return result


from collections import Counter, defaultdict


class DetectSquares:
    def __init__(self):
        self.horizontal = Counter()
        self.vertical = Counter()
        self.points = Counter()

    def add(self, point: List[int]) -> None:
        self.horizontal[point[1]] += 1
        self.vertical[point[0]] += 1
        self.points[(point[0], point[1])] += 1

    def count(self, point: List[int]) -> int:
        if point[0] not in self.vertical: return 0
        if point[1] not in self.horizontal: return 0
        total = 0
        for p, v in self.points.items():
            if p[0] == point[0] or p[1] == point[1]: continue
            if abs(point[0] - p[0]) != abs(point[1] - p[1]): continue
            total += v * self.points[(p[0], point[1])] * self.points[(point[0], p[1])]
        return total


from collections import deque
class HitCounter:
    def __init__(self):
        self.hits = deque()

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)


    def getHits(self, timestamp: int) -> int:
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len(self.hits)


from collections import Counter
class FreqStack:
    def __init__(self):
        self.val2freq = Counter()
        self.stackOfStack = []

    def push(self, val: int) -> None:
        self.val2freq[val] += 1
        f = self.val2freq[val]
        while len(self.stackOfStack) <= f:
            self.stackOfStack.append([])
        self.stackOfStack[f].append(val)

    def pop(self) -> int:
        while self.stackOfStack and len(self.stackOfStack[-1]) == 0:
            self.stackOfStack.pop(-1)
        if not self.stackOfStack: return -1
        val = self.stackOfStack[-1].pop(-1)
        self.val2freq[val] -= 1
        return val

class Node(object):
    def __init__(self, val: Optional[int] = None, children: Optional[List['Node']] = None):
        if children is None:
            children = []
        self.val = val
        self.children = children

class Codec:
    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.

        :type root: Node
        :rtype: str
        """
        if root is None: return "null:0"
        sub_strs = []
        for c in root.children:
            sub_strs.append(self.serialize(c))
        if len(sub_strs) > 0:
            return str(root.val) + ":" + str(len(root.children)) + "," + ",".join(sub_strs)
        else: return str(root.val) + ":" + str(len(root.children))


    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: Node
        """



        if data == "null:0": return None

        def build_node(s):
            arr = s.split(":")
            return Node(int(arr[0])), int(arr[1])

        nodes = data.split(",")
        def helper(start):
            node, num_child = build_node(nodes[start])
            next_index = start + 1

            for _ in range(num_child):
                child, next_index = helper(next_index)
                node.children.append(child)
            return node, next_index
        root, _ = helper(0)
        return root

    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums: return 0
        from collections import Counter
        if not nums: return 0
        nums = set(nums)
        dp = dict()

        def topo_sort(n):
            counter = Counter()
            stack = [n]
            while stack:
                node = stack.pop(-1)
                counter[node] += 1
                if counter[node] == 1:
                    stack.append(node)
                    if node - 1 in nums and node - 1 not in dp:
                        stack.append(node - 1)
                else:
                    if node - 1 in dp:
                        dp[node] = 1 + dp[node - 1]
                    else: dp[node] = 1

        for n in nums:
            if n in dp: continue
            topo_sort(n)

        return max(dp.values())

    def removeKdigits(self, num: str, k: int) -> str:
        stack = []

        for n in num:
            if stack:
                digit_n = ord(n) - ord('0')
                while stack and ord(stack[-1]) - ord('0') > digit_n and k > 0:
                    stack.pop(-1)
                    k -= 1
                stack.append(n)
            else:
                stack.append(n)

        while k > 0 and stack:
            stack.pop(-1)
            k -= 1
        result = []
        non_zero = True

        for n in stack:
            if n == '0' and non_zero: continue
            else:
                non_zero = False
                result.append(n)
        if not result: return "0"
        return "".join(result)

    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        from collections import Counter
        # import heapq
        # merged = []
        # heap = []
        # total_size = 0
        # for i, arr in enumerate(nums):
        #     if not arr: continue
        #     heapq.heappush(heap, (arr[0], 0, i))
        #     total_size += len(arr)
        # if total_size == 0: return []
        #
        # while len(merged) < total_size:
        #     n, index, i = heap[0]
        #     heapq.heappop(heap)
        #     merged.append((n, i))
        #     if index + 1 < len(nums[i]):
        #         heapq.heappush(heap, (nums[i][index + 1], index + 1, i))
        merged = []
        for i, arr in enumerate(nums):
            for n in arr:
                merged.append((n, i))
        merged.sort()

        def check(w_size):
            total_k = 0
            counter = Counter()
            left = 0
            start = 0

            while start < len(merged) and merged[start][0] - merged[left][0] <= w_size:
                k = merged[start][1]
                if counter[k] == 0:
                    total_k += 1
                counter[k] += 1
                start += 1

            if total_k == len(nums):
                return [merged[left][0], merged[start - 1][0]]

            while start < len(merged):
                k = merged[start][1]
                if counter[k] == 0:
                    total_k += 1
                counter[k] += 1

                while merged[start][0] - merged[left][0] > w_size:
                    pop_k = merged[left][1]
                    counter[pop_k] -= 1
                    if counter[pop_k] == 0:
                        total_k -= 1
                    left += 1

                start += 1
                if total_k == len(nums):
                    return [merged[left][0], merged[start - 1][0]]
            return []

        left, right = 0, merged[-1][0] - merged[0][0]
        while left <= right:
            mid = (left + right) // 2
            if not check(mid):
                left = mid + 1
            else:
                right = mid - 1
        return check(left)

    def lexicalOrder(self, n: int) -> List[int]:
        result = []

        def recur(n, prefix):
            prefix = prefix * 10
            if prefix != 0:
                for i in range(10):
                    if prefix + i > n: break
                    result.append(prefix + i)
                    recur(n, prefix + i)
            else:
                for i in range(1, 10):
                    if prefix + i > n: break
                    result.append(prefix + i)
                    recur(n, prefix + i)

        recur(n, 0)
        return result

    def decodeString(self, s: str) -> str:

        def recur(start):
            # return string, end
            result = []
            digit = ''
            i = start
            while i < len(s):
                if 0 <= ord(s[i]) - ord('0') <= 9:
                    digit += s[i]
                    i += 1
                elif s[i] == '[':
                    substr, i = recur(i + 1)
                    for _ in range(int(digit)):
                        result.append(substr)
                    digit = ''
                elif s[i] == ']':
                    return ''.join(result), i + 1
                else:
                    result.append(s[i])
                    i += 1
            return ''.join(result), i
        return recur(0)[0]

    def maximumSum(self, nums: List[int]) -> int:
        import heapq

        def get_digit_sum(n):
            total = 0
            while n != 0:
                total += n % 10
                n = n // 10
            return total

        d = dict()
        for n in nums:
            k = get_digit_sum(n)
            if k not in d:
                d[k] = [n]
            else:
                heapq.heappush(d[k], n)
                if len(d[k]) >= 3:
                    heapq.heappop(d[k])
        max_sum = -1
        for k, arr in d.items():
            if len(arr) <= 1: continue
            max_sum = max(max_sum, sum(arr))
        return max_sum


import heapq
class Twitter:
    def __init__(self):
        self.user2posts = dict()
        self.user2follows = dict()
        self.timestamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        if userId not in self.user2posts:
            self.user2posts[userId] = [(self.timestamp, tweetId)]
        else:
            self.user2posts[userId].append((self.timestamp, tweetId))
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        heap = []
        if userId not in self.user2follows:
            self.user2follows[userId] = set()
        else:
            for user in self.user2follows[userId]:
                if user not in self.user2posts or not self.user2posts[user]: continue
                arr = self.user2posts[user]
                heap.append((-arr[-1][0], arr[-1][1], user, 1))
        if userId in self.user2posts:
            arr = self.user2posts[userId]
            heap.append((-arr[-1][0], arr[-1][1], userId, 1))
        heapq.heapify(heap)
        result = []
        while len(result) < 10 and heap:
            _, tweetId, user, length = heapq.heappop(heap)
            result.append(tweetId)
            if len(self.user2posts[user]) < length + 1: continue
            arr = self.user2posts[user]
            length += 1
            heapq.heappush(heap, (-arr[-length][0], arr[-length][1], user, length))
        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId not in self.user2follows:
            self.user2follows[followerId] = {followeeId}
        else:
            self.user2follows[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId not in self.user2follows: return
        if followeeId not in self.user2follows[followerId]: return
        self.user2follows[followerId].remove(followeeId)


class SparseVector:
    def __init__(self, nums: List[int]):
        self.arr = []
        prev_zero = 0
        for n in nums:
            if n != 0:
                if prev_zero != 0:
                    self.arr.append((0, self._init_get_index(prev_zero)))
                prev_zero = 0
                self.arr.append((n, self._init_get_index(1)))
            else:
                prev_zero += 1
        if prev_zero != 0:
            self.arr.append((0, self._init_get_index(prev_zero)))

    def _init_get_index(self, numbers):
        if not self.arr: return numbers - 1
        return self.arr[-1][-1] + numbers

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        if not self.arr or not vec.arr: return 0
        index = 0
        sparse_index1 = 0
        sparse_index2 = 0
        total = 0
        while sparse_index1 < len(self.arr) and sparse_index2 < len(vec.arr):
            while sparse_index1 < len(self.arr):
                if self.arr[sparse_index1][-1] >= index: break
                sparse_index1 += 1
            while sparse_index2 < len(vec.arr):
                if vec.arr[sparse_index2][-1] >= index: break
                sparse_index2 += 1

            if sparse_index1 >= len(self.arr) or sparse_index2 >= len(vec.arr): break
            total += (self.arr[sparse_index1][0] * vec.arr[sparse_index2][0]
                      * (min(self.arr[sparse_index1][-1], vec.arr[sparse_index2][-1]) - index + 1))
            index = min(self.arr[sparse_index1][-1], vec.arr[sparse_index2][-1]) + 1
        return total


class WordDictionary:

    def __init__(self):
        self.trie = dict()
        self.END = "END"

    def addWord(self, word: str) -> None:
        t = self.trie
        for c in word:
            if c not in t:
                t[c] = dict()
            t = t[c]
        t[self.END] = None

    def search(self, word: str) -> bool:
        if not word: return False

        stack = [(self.trie, 0)]
        while stack:
            t, index = stack.pop(-1)
            if index == len(word):
                if self.END in t: return True
                continue
            if word[index] == '.':
                for k in t.keys():
                    if k == self.END: continue
                    stack.append((t[k], index + 1))
            else:
                if word[index] not in t: continue
                stack.append((t[word[index]], index + 1))
        return False

    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots1 = [(s[0], s[1]) for s in slots1]
        slots2 = [(s[0], s[1]) for s in slots2]
        slots1.sort()
        slots2.sort()
        i = 0
        index = 0
        while i < len(slots1):
            if index >= len(slots2): break
            if slots2[index][0] >= slots1[i][1]:
                i += 1
            elif slots2[index][1] <= slots1[i][0]:
                index += 1
            else:
                # slots2[index][0] < slots1[i][1]
                # slots2[index][1] > slots1[i][0]
                overlap = min(slots2[index][1], slots1[i][1]) - max(slots2[index][0], slots1[i][0])
                if overlap >= duration:
                    return [max(slots2[index][0], slots1[i][0]), max(slots2[index][0], slots1[i][0]) + duration]
                if slots2[index][1] <= slots1[i][1]:
                    index += 1
                else: i += 1
        return []

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = dict()
        END = "END"
        for w in words:
            t = trie
            for c in w:
                if c not in t: t[c] = dict()
                t = t[c]
            t[END] = None
        M = len(board)
        N = len(board[0])

        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        def gen(i, j):
            for di, dj in directions:
                if i + di < 0 or i + di >= M: continue
                if j + dj < 0 or j + dj >= N: continue
                yield i + di, j + dj

        def search(i, j, t, seen, prefix):
            if board[i][j] not in t:
                if END in t: return {prefix}
                return set()
            seen[i][j] = True
            t = t[board[i][j]]
            result = set()
            prefix += board[i][j]
            for ni, nj in gen(i, j):
                if seen[ni][nj]: continue
                r = search(ni, nj, t, seen, prefix)
                for x in r: result.add(x)
            if END in t:
                result.add(prefix)
            seen[i][j] = False
            return result

        result = set()
        for i in range(M):
            for j in range(N):
                seen = [[False for _ in range(N)] for _ in range(M)]
                r = search(i, j, trie, seen, '')
                for x in r: result.add(x)
                if len(result) == len(words): break
        return list(result)


class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        # 3 4 5 1 2 3
        # 3 1
        self.stack.append(val)
        if not self.min_stack or self.min_stack[-1][0] > val:
            self.min_stack.append((val, len(self.stack) - 1))

    def pop(self) -> None:
        if not self.stack: return
        self.stack.pop()
        while self.min_stack and self.min_stack[-1][1] >= len(self.stack):
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1][0]


class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = [(root, False)]

    def next(self) -> int:
        while self.stack:
            node, leftVisited = self.stack.pop()
            if not leftVisited:
                self.stack.append((node, True))
                if node.left is not None:
                    self.stack.append((node.left, False))
            else:
                if node.right is not None:
                    self.stack.append((node.right, False))
                return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0


class Trie:

    def __init__(self):
        self.t = dict()
        self.END = "END"

    def insert(self, word: str) -> None:
        t = self.t
        for w in word:
            if w not in t:
                t[w] = dict()
            t = t[w]
        t[self.END] = None

    def search(self, word: str) -> bool:
        t = self.t
        for w in word:
            if w not in t: return False
            t = t[w]
        return self.END in t

    def startsWith(self, prefix: str) -> bool:
        t = self.t
        for w in prefix:
            if w not in t: return False
            t = t[w]
        return True


from collections import deque
class Logger:
    def __init__(self):
        self.message2queue = dict()

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.message2queue:
            self.message2queue[message] = deque()
        q = self.message2queue[message]
        while q and timestamp - q[0] >= 10:
            q.popleft()
        if q and timestamp - q[0] < 10: return False
        q.append(timestamp)
        return True


class Vector2D:
    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.top_i = 0
        self.bot_i = 0

    def next(self) -> int:
        if self.bot_i == len(self.vec[self.top_i]):
            self.top_i += 1
            self.bot_i = 0
        self.bot_i += 1
        return self.vec[self.top_i][self.bot_i - 1]

    def hasNext(self) -> bool:
        while self.top_i < len(self.vec) and self.bot_i == len(self.vec[self.top_i]):
            self.top_i += 1
            self.bot_i = 0
        return self.top_i < len(self.vec)


class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        self.v1 = v1
        self.v2 = v2
        self.v1_index = 0
        self.v2_index = 0
        self.read_v1 = True


    def next(self) -> int:
        if (self.read_v1 and self.v1_index < len(self.v1)) \
                or ((not self.read_v1) and self.v2_index >= len(self.v2)):
            val = self.v1[self.v1_index]
            self.v1_index += 1
            self.read_v1 = False
        else:
            val = self.v2[self.v2_index]
            self.v2_index += 1
            self.read_v1 = True
        return val

    def hasNext(self) -> bool:
        return self.v1_index < len(self.v1) or self.v2_index < len(self.v2)

# class Iterator:
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self._move()

    def _move(self):
        if self.iter.hasNext():
            self.nextValue = self.iter.next()
        else:
            self.nextValue = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.nextValue

    def next(self):
        """
        :rtype: int
        """
        tmp = self.nextValue
        self._move()
        return tmp

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.nextValue is not None

def read4(buf4: List[str]) -> int:
    return 0
class Reads:
    def __init__(self):
        self.buffer = []

    def read(self, buf: List[str], n: int) -> int:
        buffer = self.buffer
        self.buffer = []

        n_read = len(buffer)
        while len(buffer) < n:
            buf4 = [''] * 4
            new_read = read4(buf4)
            n_read += new_read
            buffer.extend(buf4)
            if new_read == 0: break

        for i in range(min(n_read, n)):
            buf[i] = buffer[i]
        self.buffer = buffer[min(n_read, n):max(n_read, n)]
        return min(n_read, n)


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.M = len(matrix)
        self.N = len(matrix[0])
        self.dp = [[0 for _ in range(self.N + 1)] for _ in range(self.M + 1)]
        for i in range(self.M - 1, -1, -1):
            for j in range(self.N - 1, -1, -1):
                self.dp[i][j] = matrix[i][j] + self.dp[i + 1][j] + self.dp[i][j + 1] \
                                 - self.dp[i + 1][j + 1]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.dp[row1][col1] + self.dp[row2 + 1][col2 + 1] \
            - self.dp[row2 + 1][col1] - self.dp[row1][col2 + 1]


from collections import Counter
class TwoSum:
    def __init__(self):
        self.counter = Counter()

    def add(self, number: int) -> None:
        self.counter[number] += 1

    def find(self, value: int) -> bool:
        for k in self.counter.keys():
            if k == value - k:
                if self.counter[value - k] > 1: return True
            else:
                if self.counter[value - k] > 0: return True
        return False


class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stacks = [[nestedList, 0]]

    def _move_to_next_valid(self):
        while self.stacks:
            if self.stacks[-1][1] >= len(self.stacks[-1][0]):
                self.stacks.pop()
            else:
                if not self.stacks[-1][0][self.stacks[-1][1]].isInteger():
                    self.stacks[-1][1] += 1
                    self.stacks.append([self.stacks[-1][0][self.stacks[-1][1] - 1].getList(), 0])
                else: break

    def next(self) -> int:
        self._move_to_next_valid()
        self.stacks[-1][1] += 1
        return self.stacks[-1][0][self.stacks[-1][1] - 1].getInteger()

    def hasNext(self) -> bool:
        self._move_to_next_valid()
        return self.stacks and self.stacks[-1][1] < len(self.stacks[-1][0])


from collections import Counter
class TicTacToe:
    def __init__(self, n: int):
        self.rows = [Counter() for _ in range(n)]
        self.cols = [Counter() for _ in range(n)]
        self.diag = Counter()
        self.opp_diag = Counter()
        self.N = n

    def _check(self, counter, player):
        if counter[player] == self.N: return True
        return False

    def move(self, row: int, col: int, player: int) -> int:
        self.rows[row][player] += 1
        self.cols[col][player] += 1
        if row == col:
            self.diag[player] += 1
        if row == self.N - col - 1:
            self.opp_diag[player] += 1

        if self._check(self.rows[row], player) \
                or self._check(self.cols[col], player) \
                or self._check(self.diag, player) \
                or self._check(self.opp_diag, player):
            return player
        return 0


import random
class RandomIndex:
    def __init__(self, nums: List[int]):
        self.d = defaultdict(list)
        for i, n in enumerate(nums):
            self.d[n].append(i)

    def pick(self, target: int) -> int:
        return self.d[target][random.randint(0, len(self.d[target]) - 1)]


from collections import deque
class SnakeGame:
    def __init__(self, width: int, height: int, food: List[List[int]]):
        # 0 means empty
        # 1 means food
        # 2 means snake
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.N = height
        self.M = width
        if food:
            f = food[0]
            self.board[f[0]][f[1]] = 1
        self.snake = deque()
        self.snake.append((0, 0))
        self.board[0][0] = 2
        self.score = 0
        self.food = food
        self.food_index = 1

    def _gen_move(self, i, j, direction):
        if direction == 'R':
            return i, j + 1
        elif direction == 'D':
            return i + 1, j
        elif direction == 'U':
            return i - 1, j
        else:
            return i, j - 1

    def _is_out_of_bound_move(self, i, j):
        if i < 0 or i >= self.N: return True
        if j < 0 or j >= self.M: return True
        return False

    def move(self, direction: str) -> int:
        ni, nj = self._gen_move(self.snake[0][0], self.snake[0][1], direction)
        if self._is_out_of_bound_move(ni, nj): return -1
        if self.board[ni][nj] == 1:
            self.board[ni][nj] = 2
            self.snake.appendleft((ni, nj))
            self.score += 1
            if self.food_index < len(self.food):
                f = self.food[self.food_index]
                self.board[f[0]][f[1]] = 1
                self.food_index += 1

        else:
            tail_i, tail_j = self.snake.pop()
            self.board[tail_i][tail_j] = 0
            if self.board[ni][nj] == 2: return -1
            self.snake.appendleft((ni, nj))
            self.board[ni][nj] = 2
        return self.score





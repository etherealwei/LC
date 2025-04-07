from collections import deque
from functools import total_ordering, lru_cache
from itertools import count
from typing import *

from leetcode_0_types import *


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        index1 = m - 1
        index2 = n - 1
        index = len(nums1) - 1

        while index1 >= 0 or index2 >= 0:
            if index1 >= 0 and index2 >= 0:
                if nums1[index1] >= nums2[index2]:
                    nums1[index] = nums1[index1]
                    index1 -= 1
                else:
                    nums1[index] = nums2[index2]
                    index2 -= 1
            elif index1 >= 0:
                nums1[index] = nums1[index1]
                index1 -= 1
            else:
                nums1[index] = nums2[index2]
                index2 -= 1
            index -= 1

    def countNodes(self, root: Optional[TreeNode]) -> int:
        if root is None: return 0
        total = 0
        stack = [root]
        while stack:
            node = stack.pop(-1)
            total += 1
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        return total

    def mergeAlternately(self, word1: str, word2: str) -> str:
        result = []
        index1 = 0
        index2 = 0
        is_index1_turn = True

        while index1 < len(word1) or index2 < len(word2):
            if index1 < len(word1) and index2 < len(word2):
                if is_index1_turn:
                    result.append(word1[index1])
                    index1 += 1
                else:
                    result.append(word2[index2])
                    index2 += 1
                is_index1_turn = not is_index1_turn
            elif index1 < len(word1):
                result.append(word1[index1])
                index1 += 1
            else:
                result.append(word2[index2])
                index2 += 1
        return "".join(result)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        import math
        if root is None: return True
        stack = [(root, -math.inf, math.inf)]
        while stack:
            node, minimum, maximum = stack.pop(-1)
            if node.val <= minimum or node.val >= maximum: return False
            if node.left is not None:
                stack.append((node.left, minimum, node.val))
            if node.right is not None:
                stack.append((node.right, node.val, maximum))
        return True

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        result = []

        def recur(node, paths):
            paths.append(node)
            if node == len(graph) - 1:
                result.append([p for p in paths])
                paths.pop(-1)
                return

            for nei in graph[node]:
                recur(nei, paths)
            paths.pop(-1)

        recur(0, [])
        return result

    def minSteps(self, s: str, t: str) -> int:
        from collections import Counter
        counter_s = Counter(s)
        counter_t = Counter(t)

        total = 0
        for k, v in counter_s.items():
            diff = v - counter_t[k]
            if diff < 0: continue
            total += diff
        return total

    def numDistinct(self, s: str, t: str) -> int:
        dp = [0 for _ in range(len(t))]
        for c in s:
            for i in range(len(t) - 1, -1, -1):
                if c != t[i]: continue
                if i == 0:
                    dp[i] += 1
                else:
                    dp[i] += dp[i - 1]
        return dp[-1]

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1: return 0

        forward = [0] * len(prices)
        max_price = prices[-1]
        for i in range(len(prices) - 2, -1, -1):
            forward[i] = max(0, max_price - prices[i])
            if i + 1 < len(prices):
                forward[i] = max(forward[i], forward[i + 1])
            max_price = max(max_price, prices[i])

        backward = [0] * len(prices)
        min_price = prices[0]
        for i in range(1, len(prices)):
            backward[i] = max(0, prices[i] - min_price)
            if i - 1 >= 0:
                backward[i] = max(backward[i], backward[i - 1])
            min_price = min(min_price, prices[i])

        max_profit = 0
        for i in range(len(prices)):
            if i > 0:
                max_profit = max(max_profit, backward[i - 1] + forward[i])
            max_profit = max(max_profit, forward[i])
            max_profit = max(max_profit, backward[i])
        return max_profit

    def twoCitySchedCostDP(self, costs: List[List[int]]) -> int:
        import math
        # minimum cost if we put x ppl in city a after and including ith ppl
        dp = [[math.inf for _ in range(len(costs) // 2 + 1)] for _ in range(len(costs))]

        for i in range(len(costs) - 1, -1, -1):
            print(len(costs) - i + 2, len(costs) // 2 + 2)
            for j in range(min(len(costs) - i + 1, len(costs) // 2 + 1)):
                if i == len(costs) - 1:
                    if j == 0:
                        dp[i][j] = costs[-1][1]
                    else:
                        dp[i][j] = costs[-1][0]
                else:
                    if j == 0:
                        dp[i][j] = costs[i][1] + dp[i + 1][j]
                    else:
                        dp[i][j] = min(costs[i][1] + dp[i + 1][j], costs[i][0] + dp[i + 1][j - 1])
        return dp[0][-1]

    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        savings = [(costs[i][0] - costs[i][1], i) for i in range(len(costs))]
        savings.sort()
        total = 0
        for i in range(len(costs)):
            index = savings[i][1]
            if i < len(costs) // 2:
                total += costs[index][0]
            else:
                total += costs[index][1]
        return total

    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None: return head
        if head.next is None: return head

        odd_head = head
        odd_tail = head
        even_head = head.next
        even_tail = head.next
        next_node = even_tail.next
        odd_tail.next = None
        even_tail.next = None

        append_odd = True
        while next_node is not None:
            current_node = next_node
            next_node = next_node.next
            current_node.next = None
            if append_odd:
                odd_tail.next = current_node
                odd_tail = current_node
            else:
                even_tail.next = current_node
                even_tail = current_node
            append_odd = not append_odd
        odd_tail.next = even_head
        return odd_head

    def getKth(self, lo: int, hi: int, k: int) -> int:
        dp = dict()

        def getPower(n):
            if n == 1: return 1
            if n in dp: return dp[n]
            if n % 2 == 0:
                p = 1 + getPower(n // 2)
            else:
                p = 1 + getPower(3 * n + 1)
            dp[n] = p
            return p

        nums = [(getPower(n), n) for n in range(lo, hi + 1)]
        nums.sort()
        return nums[k - 1][1]

    def tupleSameProduct(self, nums: List[int]) -> int:
        m = dict()

        for i in range(len(nums)):
            for j in range(i):
                k = nums[i] * nums[j]
                if k not in m:
                    m[k] = {(i, j)}
                else:
                    m[k].add((i, j))

        total = 0
        for k, s in m.items():
            if len(s) < 2: continue
            total += 8 * (len(s) * (len(s) - 1)) // 2
        return total

    def beautySum(self, s: str) -> int:
        from collections import Counter
        total = 0
        for i in range(1, len(s)):
            counter = Counter()
            counter[s[i]] += 1
            for j in range(i - 1, -1, -1):
                counter[s[j]] += 1
                min_freq = min(counter.values())
                max_freq = max(counter.values())
                total += max_freq - min_freq
        return total

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        from collections import Counter
        counter = Counter(nums)
        heap = [(-v, k) for k, v in counter.items()]
        heapq.heapify(heap)

        result = []
        while len(result) < k:
            result.append(heap[0][1])
            heapq.heappop(heap)
        return result

    def findLaddersTLE(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordSet = set(wordList)
        if endWord not in wordSet: return []

        seen = dict()
        queue = deque()
        queue.append((beginWord, []))

        while queue:
            node, paths = queue.popleft()
            new_path = paths + [node]
            if node in seen:
                if len(seen[node][0]) == len(new_path):
                    seen[node].append(new_path)
                else:
                    continue
            else:
                seen[node] = [new_path]
            if node == endWord: continue

            for i in range(len(node)):
                for j in range(26):
                    c = chr(ord('a') + j)
                    neighbor = node[:i] + c + node[i + 1:]
                    if neighbor not in wordSet or neighbor == node: continue
                    queue.append((neighbor, [p for p in new_path]))
        if endWord not in seen: return []
        return seen[endWord]

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordSet = set(wordList)
        if endWord not in wordSet: return []
        graph = dict()

        def gen(node):
            for i in range(len(node)):
                for j in range(26):
                    c = chr(ord('a') + j)
                    neighbor = node[:i] + c + node[i + 1:]
                    if neighbor not in wordSet or neighbor == node: continue
                    yield neighbor

        min_steps = dict()
        queue = deque()
        queue.append((beginWord, 0))
        min_steps[beginWord] = 0

        while queue:
            node, steps = queue.popleft()
            neighbors = []
            for nei in gen(node):
                neighbors.append(nei)
                if nei in min_steps: continue
                min_steps[nei] = steps + 1
                queue.append((nei, steps + 1))
            graph[node] = neighbors

        all_paths = []

        def DFS(node, paths):
            if len(paths) > min_steps[node]: return
            paths.append(node)
            if node == endWord:
                all_paths.append([p for p in paths])
                paths.pop(-1)
                return

            for nei in graph[node]:
                DFS(nei, paths)
            paths.pop(-1)

        DFS(beginWord, [])
        return all_paths

    def minCut(self, s: str) -> int:
        # "a|aba|b" "aa|bab"

        cut_dp = [i for i in range(len(s))]
        p_dp = [[False for _ in range(len(s))] for _ in range(len(s))]

        for i in range(len(s)):
            for j in range(i + 1):
                if i == j:
                    p_dp[j][i] = True
                elif s[i] == s[j]:
                    if i == j + 1:
                        p_dp[j][i] = True
                    else:
                        p_dp[j][i] = p_dp[j + 1][i - 1]
                else:
                    p_dp[j][i] = False

        for i in range(1, len(s)):
            for j in range(i + 1):
                if p_dp[j][i]:
                    if j == 0:
                        cut_dp[i] = 0
                        break
                    else:
                        cut_dp[i] = min(cut_dp[i], 1 + cut_dp[j - 1])
        return cut_dp[-1]

    def shortestPalindrome(self, s: str) -> str:
        if not s: return ""

        def check(length):
            if length == 1: return True
            stack = []
            i = 0
            while i < length // 2:
                stack.append(s[i])
                i += 1
            if length % 2 == 1:
                i += 1
            while i < length:
                if s[i] != stack[-1]: return False
                stack.pop(-1)
                i += 1
            return True

        for i in range(len(s), 0, -1):
            if check(i):
                return ''.join(reversed(s[i:])) + s

    def removeOccurrences(self, s: str, part: str) -> str:
        def getKmp(s):
            kmp = [0] * len(s)
            for i in range(1, len(s)):
                c = s[i]
                pos = kmp[i - 1]
                while pos != 0 and c != s[pos]:
                    pos = kmp[pos - 1]
                if c == s[pos]:
                    kmp[i] = pos + 1
            return kmp

        kmp = getKmp(part)
        stack = []

        index = 0
        for c in s:
            while index != 0 and c != part[index]:
                index = kmp[index - 1]
            if c == part[index]:
                index += 1
            stack.append((c, index))

            if index == len(part):
                for _ in range(len(part)): stack.pop(-1)
                if stack:
                    index = stack[-1][1]
                else:
                    index = 0
        return ''.join([c[0] for c in stack])

    def find_words(self, words, letters):
        from collections import Counter
        letters_counter = Counter(letters)
        dp = dict()

        def recur(i, counter):
            key = [str(k) + ":" + str(v) for k, v in counter.items()]
            key = ",".join(key)
            if (i, key) in dp: return dp[(i, key)]
            if i >= len(words): return []

            new_counter = {k: v for k, v in counter.items()}
            word = words[i]
            matched = True
            for c in word:
                if c not in new_counter or new_counter[c] == 0:
                    matched = False
                    break
                new_counter[c] -= 1

            result1 = recur(i + 1, counter)
            result2 = []
            if matched:
                result2 = [word] + recur(i + 1, new_counter)
            if len(result1) > len(result2):
                dp[(i, key)] = result1
            else:
                dp[(i, key)] = result2
            return dp[(i, key)]

        return recur(0, letters_counter)

    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        N = len(flights)
        K = len(days[0])
        for i in range(N):
            flights[i][i] = 1

        dp = [[0 for _ in range(N)] for _ in range(K)]
        for ik in range(K - 1, -1, -1):
            for i in range(N):
                dp[ik][i] = days[i][ik]
                if ik != K - 1:
                    dep = [days[i][ik] + dp[ik + 1][j] for j in range(N) if flights[i][j] == 1]
                    if dep:
                        dp[ik][i] = max(dp[ik][i], max(dep))
        dep = []
        for i in range(N):
            if flights[0][i] == 1: dep.append(dp[0][i])
        return max(dep)

    # def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
    #     import math
    #     M = len(dungeon)
    #     N = len(dungeon[0])
    #     dp = [[math.inf for _ in range(N)] for _ in range(M)]
    #
    #     def gen(i, j):
    #         dirs = [(1, 0), (0, 1)]
    #         for di, dj in dirs:
    #             if i + di < 0 or i + di >= M: continue
    #             if j + dj < 0 or j + dj >= N: continue
    #             yield i + di, j + dj
    #
    #     for i in range(M - 1, -1, -1):
    #         for j in range(N - 1, -1, -1):
    #             if i == M - 1 and j == N - 1:
    #                 dp[i][j] = max(1, 1 -dungeon[i][j])
    #                 continue
    #
    #             for ni, nj in gen(i, j):
    #                 # X >= 1
    #                 # X + dungeon[i][j] >= 1
    #                 # X + dungeon[i][j] >= dp[ni][nj]
    #                 dp[i][j] = min(dp[i][j], max(1, dp[ni][nj] - dungeon[i][j]))
    #     return dp[0][0]

# def maxProfitTLE(self, k: int, prices: List[int]) -> int:
#      dp = [[0 for _ in range(k + 1)] for _ in range(len(prices))]
#      for i in range(len(prices) - 2, -1, -1):
#          min_prices = prices[i]
#          profit = 0
#          for j in range(i + 1, len(prices)):
#              profit = max(profit, prices[j] - min_prices)
#              for sub_k in range(1, k + 1):
#                  if j + 1 < len(prices):
#                      dp[i][sub_k] = max(dp[i][sub_k], dp[j + 1][sub_k], profit + dp[j + 1][sub_k - 1])
#                  dp[i][sub_k] = max(dp[i][sub_k], profit)
#              min_prices = min(min_prices, prices[j])
#      return dp[0][-1]
#
# def maxProfit(self, k: int, prices: List[int]) -> int:
#     dp = [[0 for _ in range(k + 1)] for _ in range(len(prices))]
#     for sub_k in range(1, k + 1):
#         debt = prices[-1]
#         for i in range(len(prices) - 2, -1, -1):
#             dp[i][sub_k] = max(dp[i][sub_k], dp[i + 1][sub_k], debt - prices[i])
#            debt = max(debt, dp[i + 1][sub_k - 1] + prices[i])
#     return dp[0][-1]
#
#
#  def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
#      levels = []
#      if not root: return levels
#
#      from collections import deque
#      queue = deque()
#      queue.append((root, 0))
#
#      while queue:
#          node, level = queue.popleft()
#          while len(levels) <= level:
#              levels.append([])
#          levels[level].append(node.val)
#          if node.left is not None:
#              queue.append((node.left, level + 1))
#          if node.right is not None:
#              queue.append((node.right, level + 1))
#      return levels
#
#
#  def queryResults(self, limit: int, queries: List[List[int]]) -> List[int]:
#      from collections import Counter
#      total = 0
#      balls = dict()
#      counter = Counter()
#
#      result = []
#      for q in queries:
#          if q[0] not in balls:
#              balls[q[0]] = 0
#          prev_color = balls[q[0]]
#          new_color = q[1]
#          balls[q[0]] = new_color
#          if prev_color == new_color:
#              result.append(total)
#              continue
#          if prev_color == 0:
#              if counter[new_color] == 0:
#                  total += 1
#              counter[new_color] += 1
#              result.append(total)
#          else:
#              counter[prev_color] -= 1
#              if counter[prev_color] == 0:
#                  del counter[prev_color]
#                  total -= 1
#              if counter[new_color] == 0:
#                  total += 1
#              counter[new_color] += 1
#              result.append(total)
#      return result
#
#
#  def flatten(self, head):
#      def recur(head):
#          node = head
#          prev = None
#
#          while node is not None:
#              if node.child is not None:
#                  child_head, child_tail = recur(node.child)
#                  child_tail.next = node.next
#                  node.next = child_head
#                  prev = child_tail
#                  node = prev.next
#              else:
#                  prev = node
#                  node = node.next
#          return node, prev
#      return recur(head)[0]

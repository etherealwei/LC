from typing import *

from leetcode_0_types import *

class Solution:
    def addOperators_bruteforce(self, num: str, target: int) -> List[str]:
        result = []
        for i, c in enumerate(num):
            new_result = []
            if i == 0:
                new_result.append(c)
            else:
                for r in result:
                    new_result.append(r + "+" + c)
                    new_result.append(r + "-" + c)
                    new_result.append(r + "*" + c)
                    new_result.append(r + c)
            result = new_result
        final_results = []

        def check(r):
            has_digit = False
            zero_count = 0
            for c in r:
                if c == '-' or c == '+' or c == '*':
                    has_digit = False
                    zero_count = 0
                elif c != '0':
                    has_digit = True
                if c == '0':
                    zero_count += 1
                    if not has_digit and zero_count > 1: return False
            return True

        for r in result:
            if r[-1] == "+" or r[-1] == '-' or r[-1] == '*': continue
            if not check(r): continue
            try:
                if eval(r) == target: final_results.append(r)
            except: pass
        return final_results

    def alienOrder(self, words: List[str]) -> str:
        class Trie:
            def __init__(self):
                self.ordered_keys = []
                self.dict = dict()

            def add(self, w, i):
                if i >= len(w):
                    return len(self.dict) == 0
                self.ordered_keys.append(w[i])
                if w[i] not in self.dict:
                    self.dict[w[i]] = Trie()
                return self.dict[w[i]].add(w, i + 1)

        trie = Trie()
        for w in words:
            if not trie.add(w, 0): return ""

        graph = dict()
        def populate_graph(trie: Trie):
            for i in range(len(trie.ordered_keys)):
                if trie.ordered_keys[i] not in graph:
                    graph[trie.ordered_keys[i]] = set()
                for j in range(i + 1, len(trie.ordered_keys)):
                    if trie.ordered_keys[j] == trie.ordered_keys[j - 1]: continue
                    graph[trie.ordered_keys[i]].add(trie.ordered_keys[j])

            for t in trie.dict.values():
                populate_graph(t)

        populate_graph(trie)
        from collections import deque
        queue = deque()

        def dfs(node): # returns if has circle
            for nei in graph[node]:
                if nei in seen:
                    continue
                seen.add(nei)
                dfs(nei)
            queue.appendleft(node)

        def check_circle(node, seen, paths):
            seen.add(node)
            paths.add(node)
            for nei in graph[node]:
                if nei in paths: return True
                if nei in seen: continue
                if check_circle(nei, seen, paths): return True
            paths.remove(node)
            return False


        seen = set()
        for n in graph.keys():
            if check_circle(n, set(), set()):
                return ""
            if n in seen: continue
            seen.add(n)
            dfs(n)
        return "".join(queue)

    def countPalindromesTLE(self, s: str) -> int:
        mod = 10 ** 9 + 7
        dp = [[[0 for _ in range(6)] for _ in range(len(s))] for _ in range(len(s))]
        for k in range(1, len(s) + 1):
            for i in range(len(s) - k + 1):
                j = i + k - 1
                if i == j:
                    dp[i][j][1] = 1
                    continue
                if j - i == 1:
                    if s[i] == s[j]:
                        dp[i][j] = [0, 2, 1, 0, 0, 0]
                    else:
                        dp[i][j] = [0, 2, 0, 0, 0, 0]
                else:
                    if s[i] == s[j]:
                        for n in range(1, 4):
                            dp[i][j][n + 2] += dp[i + 1][j - 1][n]
                    for n in range(1, 6):
                        dp[i][j][n] += dp[i + 1][j][n] - dp[i + 1][j - 1][n]
                        dp[i][j][n] += dp[i][j - 1][n] - dp[i + 1][j - 1][n]
                        dp[i][j][n] += dp[i + 1][j - 1][n]
                        dp[i][j][n] %= mod
        return dp[0][-1][5]

    def countPalindromes(self, s: str) -> int:
        mod = 10 ** 9 + 7
        prefix = [[0 for _ in range(100)] for _ in range(len(s))]
        suffix = [[0 for _ in range(100)] for _ in range(len(s))]

        dp = [0 for _ in range(10)]
        for i in range(len(s)):
            for x in range(len(dp)):
                prefix[i][int(str(x) + s[i])] += dp[x]
            if i > 1:
                for x in range(100):
                    prefix[i][x] += prefix[i - 1][x]
            dp[int(s[i])] += 1

        dp = [0 for _ in range(10)]
        for i in range(len(s) - 1, -1, -1):
            for x in range(len(dp)):
                suffix[i][int(str(x) + s[i])] += dp[x] + suffix[i - 1][int(str(x) + s[i])]
            if i < len(s) - 1:
                for x in range(100):
                    suffix[i][x] += suffix[i + 1][x]
            dp[int(s[i])] += 1

        result = 0
        for i in range(2, len(s) - 2):
            for x in range(100):
                result = (result + prefix[i - 1][x] * suffix[i + 1][x]) % mod
        return result

    def minRemoveToMakeValid(self, s: str) -> str:
        is_valid = [False for _ in range(len(s))]
        stack = []
        for i, c in enumerate(s):
            if c != '(' and c != ')':
                is_valid[i] = True
            elif c == '(':
                stack.append(i)
            else:
                if stack:
                    is_valid[stack[-1]] = True
                    stack.pop(-1)
                    is_valid[i] = True
        return ''.join([s[i] for i in range(len(s)) if is_valid[i]])

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import deque
        if not root: return []
        order = dict()

        queue = deque()
        queue.append((root, 0))
        while queue:
            node, col = queue.popleft()
            if col not in order:
                order[col] = [node.val]
            else:
                order[col].append(node.val)
            if node.left is not None:
                queue.append((node.left, col - 1))
            if node.right is not None:
                queue.append((node.right, col + 1))
        result = []
        for k in sorted(order.keys()):
            result.append(order[k])
        return result

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList.append(beginWord)
        wordList = set(wordList)
        if endWord not in wordList: return 0
        graph = dict()
        characters = [chr(i + ord('a')) for i in range(26)]
        for w in wordList:
            if w not in graph: graph[w] = set()
            for i in range(len(w)):
                for c in characters:
                    if c == w[i]: continue
                    new_w = w[:i] + c + w[i + 1:]
                    if new_w not in wordList: continue
                    graph[w].add(new_w)

        from collections import deque
        queue = deque()
        queue.append((beginWord, 1))
        seen = {beginWord}
        while queue:
            node, step = queue.popleft()
            for nei in graph[node]:
                if nei in seen: continue
                if nei == endWord: return step + 1
                seen.add(nei)
                queue.append((nei, step + 1))
        return 0

    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        intermediate = []
        index_1, index_2 = 0, 0
        leftover_1, leftover_2 = encoded1[0][1], encoded2[0][1]
        while True:
            if leftover_1 == 0:
                index_1 += 1
                if index_1 >= len(encoded1): break
                leftover_1 = encoded1[index_1][1]
            num_1 = encoded1[index_1][0]
            if leftover_2 == 0:
                index_2 += 1
                if index_2 >= len(encoded2): break
                leftover_2 = encoded2[index_2][1]
            num_2 = encoded2[index_2][0]
            intermediate.append([num_1 * num_2, min(leftover_1, leftover_2)])
            r = min(leftover_2, leftover_1)
            leftover_1 -= r
            leftover_2 -= r
        result = []
        for inter in intermediate:
            if result and inter[0] == result[-1][0]:
                result[-1][1] += inter[1]
            else:
                result.append(inter)
        return result

    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        matrix = [[-1 for _ in range(n)] for _ in range(m)]
        if not head: return matrix
        # n, m - 1, n - 1, m - 2, n - 2, m - 3, n - 3, etc
        class Iterator:
            def __init__(self):
                self.dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                self.start = 0

            def turn(self):
                self.start = (self.start + 1) % len(self.dirs)

            def next(self, x, y):
                return x + self.dirs[self.start][0], y + self.dirs[self.start][1]

        iterator = Iterator()
        x, y = 0, 0
        matrix[0][0] = head.val
        node = head.next
        for _ in range(n - 1):
            if node is None: break
            x, y = iterator.next(x, y)
            matrix[x][y] = node.val
            node = node.next

        iterator.turn()
        iteration = 1
        while n - iteration >= 0 and m - iteration >= 0:
            if node is None: break
            for _ in range(m - iteration):
                if node is None: break
                x, y = iterator.next(x, y)
                matrix[x][y] = node.val
                node = node.next
            iterator.turn()
            for _ in range(n - iteration):
                if node is None: break
                x, y = iterator.next(x, y)
                matrix[x][y] = node.val
                node = node.next
            iterator.turn()
            iteration += 1
        return matrix

    def findShortestPath(self, master: 'GridMaster') -> int:
        reverse_steps = {
            'L': 'R',
            'R': 'L',
            'U': 'D',
            'D': 'U'
        }

        steps_to_axis = {
            'L': (-1, 0),
            'R': (1, 0),
            'U': (0, 1),
            'D': (0, -1)
        }

        seen = set()
        target = None
        def dfs(x, y, paths):
            nonlocal target
            if master.isTarget():
                target = (x, y)
            seen.add((x, y))
            for direction in steps_to_axis.keys():
                if master.canMove(direction):
                    new_x, new_y = x + steps_to_axis[direction][0], y + steps_to_axis[direction][1]
                    if (new_x, new_y) in seen: continue
                    master.move(direction)
                    paths.append(direction)
                    dfs(new_x, new_y, paths)
                    paths.pop(-1)
            if paths:
                master.move(reverse_steps[paths[-1]])

        dfs(0, 0, [])
        if target is None: return -1
        from collections import deque
        queue = deque()
        queue.append((0, 0, 0))
        seen2 = {(0, 0)}
        while queue:
            x, y, steps = queue.popleft()
            for direction in steps_to_axis.keys():
                new_x, new_y = x + steps_to_axis[direction][0], y + steps_to_axis[direction][1]
                if new_x == target[0] and new_y == target[1]: return steps + 1

                if (new_x, new_y) not in seen: continue
                if (new_x, new_y) in seen2: continue
                seen2.add((new_x, new_y))
                queue.append((new_x, new_y, steps + 1))
        return -1

    def isPossible(self, n: int, edges: List[List[int]]) -> bool:
        graph = [set() for _ in range(n)]
        for e in edges:
            graph[e[0] - 1].add(e[1] - 1)
            graph[e[1] - 1].add(e[0] - 1)

        odd_nodes = []
        for i in range(n):
            if len(graph[i]) % 2 == 1: odd_nodes.append(i)

        if len(odd_nodes) >= 5: return False
        if len(odd_nodes) == 0: return True
        if len(odd_nodes) == 1: return False

        if len(odd_nodes) == 2:
            if odd_nodes[1] not in graph[odd_nodes[0]]: return True
            else:
                # cant add an existing edge
                for i in range(n):
                    if i == odd_nodes[0] or i == odd_nodes[1]: continue
                    if odd_nodes[0] not in graph[i] and odd_nodes[1] not in graph[i]: return True
                return False
        if len(odd_nodes) == 3: return False
        # odd nodes are 4
        possibles = {odd_nodes[1], odd_nodes[2], odd_nodes[3]}
        for i in range(1, 4):
            possibles.remove(odd_nodes[i])
            if odd_nodes[0] not in graph[odd_nodes[i]]:
                remains = list(possibles)
                if remains[0] not in graph[remains[1]]: return True
            possibles.add(odd_nodes[i])
        return False

    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        sums = [0] * n
        for r in roads:
            sums[r[0]] += 1
            sums[r[1]] += 1
        sums.sort()
        return sum(sums[i] * (i + 1) for i in range(n))

    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        if (maxChoosableInteger + 1) * (maxChoosableInteger) / 2 < desiredTotal: return False
        seen = dict()
        def dfs(mask, desiredTotal):
            if (mask, desiredTotal) in seen: return seen[mask, desiredTotal]
            result = False
            for i in range(1, maxChoosableInteger + 1):
                if (mask & (1 << i)) != 0: continue
                if i >= desiredTotal:
                    result = True
                    break

                new_mask = (mask | (1 << i))
                result = not dfs(new_mask, desiredTotal - i)
                if result: break
            seen[mask, desiredTotal] = result
            return result
        r = dfs(0, desiredTotal)
        return r

    def minInsertionsTLE(self, s: str) -> int:
        import math
        def get_order(i):
            return ord(s[i]) - ord('a')

        # for subarray s[i:j + 1], minimum insertion to make it surrounded by character a.
        dp = [[[math.inf for _ in range(26)] for _ in range(len(s))] for _ in range(len(s))]
        for k in range(1, len(s) + 1):
            for i in range(len(s) - k + 1):
                j = i + k - 1
                if i == j:
                    for a in range(26):
                        if a == get_order(i):
                            dp[i][j][a] = 0
                        else:
                            dp[i][j][a] = 2
                elif k == 2:
                    if s[i] == s[j]:
                        for a in range(26):
                            if a == get_order(i):
                                dp[i][j][a] = 0
                            else:
                                dp[i][j][a] = 2
                    else:
                        for a in range(26):
                            if a == get_order(i) or a == get_order(j):
                                dp[i][j][a] = 1
                            else:
                                dp[i][j][a] = 3
                else:
                    if s[i] == s[j]:
                        min_middle = min(dp[i + 1][j - 1][:])
                        for a in range(26):
                            if a == get_order(i):
                                dp[i][j][a] = min_middle
                            else:
                                dp[i][j][a] = 2 + min_middle
                    else:
                        min_right = min(dp[i + 1][j][:])
                        min_left = min(dp[i][j - 1][:])
                        for a in range(26):
                            if a == get_order(i):
                                dp[i][j][a] = 1 + min_right
                            elif a == get_order(j):
                                dp[i][j][a] = 1 + min_left
                            else:
                                dp[i][j][a] = min(3 + min_left, 3 + min_right)
        return min(dp[0][-1][:])


    def minInsertions(self, s: str) -> int:
        import math
        def get_order(i):
            return ord(s[i]) - ord('a')

        # for subarray s[i:j + 1], minimum insertion to make it surrounded by character a.
        dp = [[math.inf for _ in range(len(s))] for _ in range(len(s))]
        for k in range(1, len(s) + 1):
            for i in range(len(s) - k + 1):
                j = i + k - 1
                if i == j:
                    dp[i][j] = 0
                elif k == 2:
                    if s[i] == s[j]:
                        dp[i][j] = 0
                    else:
                        dp[i][j] = 1
                else:
                    if s[i] == s[j]:
                        dp[i][j] = dp[i + 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1]


    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        tree = [[] for _ in range(n)]
        for e in edges:
            tree[e[0]].append(e[1])
            tree[e[1]].append(e[0])

        counts = [[0 for _ in range(26)] for _ in range(n)]
        seen = [0 for _ in range(n)]
        stack = [0]
        while stack:
            node = stack.pop(-1)
            if seen[node] == 1: #traversing second time
                counts[node][ord(labels[node]) - ord('a')] += 1
                for i in range(26):
                    for nei in tree[node]:
                        if seen[nei] == 1: continue
                        counts[node][i] += counts[nei][i]
                seen[node] += 1
            else:
                # traversing first time
                seen[node] += 1
                stack.append(node)
                for nei in tree[node]:
                    if seen[nei] == 1: continue
                    stack.append(nei)
        return [counts[i][ord(labels[i]) - ord('a')] for i in range(n)]

    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        from collections import Counter
        lines = dict()
        mod = 10 ** 9 + 7
        for r in rectangles:
            x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
            if min(x1, x2) not in lines:
                lines[min(x1, x2)] = []
            lines[min(x1, x2)].append([(min(y1, y2), max(y1, y2)), '+'])
            if max(x1, x2) not in lines:
                lines[max(x1, x2)] = []
            lines[max(x1, x2)].append([(min(y1, y2), max(y1, y2)), '-'])

        def calculate_length(segments):
            counter = Counter()
            for min_y, max_y in segments:
                counter[min_y] += 1
                counter[max_y] -= 1
            length = 0
            prev_y = None
            running = 0
            for y in sorted(counter.keys()):
                if prev_y is not None:
                    if running > 0:
                        length += y - prev_y
                running += counter[y]
                prev_y = y
            return length


        segments = Counter()
        prev_x = None
        area = 0
        for x in sorted(lines.keys()):
            if prev_x is not None:
                area = (area + (x - prev_x) * calculate_length([k for k, v in segments.items() if v != 0])) % mod
            for s in lines[x]:
                if s[-1] == '-':
                    segments[s[0]] -= 1
                else:
                    segments[s[0]] += 1
            prev_x = x
        return area

    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        # result = []
        # M = len(nums)
        # N = max([len(n) for n in nums])
        # for s in range(M + N - 1):
        #     for i in range(min(s, M - 1), -1, -1):
        #         if i <= s - len(nums[i]): continue
        #         j = s - i # s - i < len(nums[i]) -> i > s - len(nums[i])
        #         result.append(nums[i][j])
        # return result
        rows = []
        for i in range(len(nums) - 1, -1, -1):
            for j in range(len(nums[i])):
                s = i + j
                while len(rows) <= s:
                    rows.append([])
                rows[i + j].append(nums[i][j])
        result = []
        for r in rows:
            result.extend(r)
        return result

    def maxDifferenceTLE(self, s: str, k: int) -> int:
        freqs = [[0 for _ in range(5)] for _ in range(len(s))]
        for i in range(len(s)):
            if i == 0:
                freqs[i][int(s[i])] += 1
            else:
                freqs[i][int(s[i])] += 1
                for j in range(5):
                    freqs[i][j] += freqs[i - 1][j]

        def check(freq):
            max_odd = None
            min_even = None
            for f in freq:
                if f % 2 == 1:
                    if max_odd is None: max_odd = f
                    else: max_odd = max(max_odd, f)
                elif f != 0:
                    if min_even is None: min_even = f
                    else: min_even = min(min_even, f)
            if max_odd is None or min_even is None: return None
            return max_odd - min_even


        max_result = None
        for i in range(k - 1, len(s)):
            for j in range(i - k + 2):
                if j == 0:
                    r = check(freqs[i])
                else:
                    freq = [0] * 5
                    for x in range(5):
                        freq[x] = freqs[i][x] - freqs[j - 1][x]
                    r = check(freq)
                if r is not None:
                    if max_result is None: max_result = r
                    else: max_result = max(max_result, r)
        return max_result

    def minOperations(self, n: int) -> int:
        if n == 1: return 0
        # 1 3 5 7 9
        median_index = n // 2
        # 4, 2
        result = (2 + 2 * (n // 2)) * (n // 2) / 2
        if n % 2 == 0:
            # 1 3 5 7 9 11
            result -= median_index
        return int(result)

    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for r in relations:
            graph[r[1] - 1].append(r[0] - 1)
        visit_counts = [0] * n
        min_start = [0] * n

        def populate(start_node):
            stack = [start_node]
            while stack:
                node = stack.pop(-1)
                if visit_counts[node] == 0:
                    visit_counts[node] += 1
                    stack.append(node)
                    for nei in graph[node]:
                        if visit_counts[nei] == 2: continue
                        stack.append(nei)
                else:
                    visit_counts[node] += 1
                    for nei in graph[node]:
                        min_start[node] = max(min_start[node], min_start[nei] + time[nei])

        for i in range(n):
            if visit_counts[i] != 2:
                populate(i)
        return max(min_start[i] + time[i] for i in range(n))

    def minSwapsCouples(self, row: List[int]) -> int:
        graph = [[] for _ in range(len(row) // 2)]

        for i in range(len(row)):
            for j in range(i):
                if max(row[i], row[j]) - min(row[i], row[j]) == 1 and min(row[i], row[j]) % 2 == 0:
                    graph[i // 2].append(j // 2)
                    graph[j // 2].append(i // 2)
                    break

        seen = [False] * (len(row) // 2)
        def size_of_connected_component(node):
            stack = [node]
            seen[node] = True
            count = 1
            while stack:
                n = stack.pop(-1)
                for nei in graph[n]:
                    if seen[nei]: continue
                    seen[nei] = True
                    count += 1
                    stack.append(nei)
            return count
        result = 0
        for i in range(len(graph)):
            result += size_of_connected_component(i) - 1
        return result

    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        from collections import Counter
        def topological_sort(graph, start_node, visited):
            order = []
            stack = [start_node]
            visit_counts = Counter()
            while stack:
                node = stack.pop(-1)
                if visit_counts[node] == 0:
                    visit_counts[node] += 1
                    stack.append(node)
                    for nei in graph[node]:
                        if visit_counts[nei] == 2:
                            continue
                        elif visit_counts[nei] == 1:
                            # circle founded
                            return None
                        stack.append(nei)
                else:
                    visit_counts[node] += 1
                    if not visited[node]:
                        order.append(node)
                        visited[node] = True
            return order

        for i in range(len(group)):
            if group[i] == -1:
                group[i] = m
                m += 1
        group_graph = [set() for _ in range(m)]
        for i, items in enumerate(beforeItems):
            for item in items:
                if group[i] == group[item]:
                    # no self edge
                    continue
                group_graph[group[i]].add(group[item])

        visited = [False] * m
        group_order = []
        for i in range(m):
            if visited[i]: continue
            new_order = topological_sort(group_graph, i, visited)
            if new_order is None:
                return []
            group_order.extend(new_order)
        group_order = {k:i for i, k in enumerate(group_order)}

        in_group_graph = [dict() for _ in range(m)]
        for i, items in enumerate(beforeItems):
            for item in items:
                if group[i] != group[item]: continue
                if i not in in_group_graph[group[i]]:
                    in_group_graph[group[i]][i] = [item]
                else:
                    in_group_graph[group[i]][i].append(item)
        for i in range(n):
            if i not in in_group_graph[group[i]]:
                in_group_graph[group[i]][i] = []

        in_group_order = []
        for g in range(m):
            seen = {k: False for k in in_group_graph[g].keys()}
            total_order = []
            for i in in_group_graph[g].keys():
                if seen[i]: continue
                new_order = topological_sort(in_group_graph[g], i, seen)
                if new_order is None:
                    return []
                total_order.extend(new_order)
            in_group_order.append({node: i for i, node in enumerate(total_order)})

        elements = [(group_order[group[i]], in_group_order[group[i]][i], i) for i in range(n)]
        elements.sort()
        return [e[-1] for e in elements]

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import Counter
        counter = Counter()
        left = 0
        right = 0 # exclusive

        while right < len(nums):
            counter[nums[right]] += 1
            if counter[nums[right]] > k:
                counter[nums[right]] -= 1
                break
            right += 1

        max_result = right - left
        while right < len(nums):
            counter[nums[right]] += 1
            right += 1
            while counter[nums[right - 1]] > k and left < right:
                counter[nums[left]] -= 1
                left += 1
            max_result = max(max_result, right - left)
        return max_result

    def mySqrt(self, x: int) -> int:
        def check(y):
            return y ** 2 > x

        left, right = 0, x
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right

    def minOperations(self, nums: List[int], x: int, y: int) -> int:
        nums.sort(reverse=True)
        def check(ops):
            running_ops = ops
            for n in nums:
                if n <= ops * y: break
                count = 0
                while True:
                    count += 1
                    running_ops -= 1
                    if running_ops < 0: return False
                    if n <= count * x + (ops - count) * y:
                        break
            return True

        max_number = max(nums)
        left, right = max_number // x, (max_number // y) + 1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left

    def countMaxOrSubsets(self, nums: List[int]) -> int:
        max_or = 0
        for n in nums:
            max_or = max_or | n

        dp = dict()
        def recur(i, value):
            nonlocal max_or
            if (i, value) in dp: return dp[i, value]
            count = 0
            if i + 1 < len(nums):
                count += recur(i + 1, value)
                count += recur(i + 1, value | nums[i])
            else:
                if value == max_or: count += 1
                if (value | nums[i]) == max_or: count += 1
            dp[i, value] = count
            return count
        return recur(0, 0)

    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        import heapq
        import math
        M = len(matrix)
        N = len(matrix[0])

        answer = [[-1 for _ in range(N)] for _ in range(M)]

        groups = [[(i, j, 1) for j in range(N)] for i in range(M)]
        def find(i, j):
            if groups[i][j][0] == i and groups[i][j][1] == j: return (i, j)
            else:
                g = find(groups[i][j][0], groups[i][j][1])
                groups[i][j] = (g[0], g[1], groups[i][j][2])
                return g

        def union(i1, j1, i2, j2):
            g1 = find(i1, j1)
            g2 = find(i2, j2)
            if groups[g1[0]][g1[1]][2] > groups[g2[0]][g2[1]][2]:
                groups[g2[0]][g2[1]] = (g1[0], g1[1], groups[g2[0]][g2[1]][2])
                groups[g1[0]][g1[1]] = (g1[0], g1[1], groups[g1[0]][g1[1]][2] + groups[g2[0]][g2[1]][2])
            else:
                groups[g1[0]][g1[1]] = (g2[0], g2[1], groups[g1[0]][g1[1]][2])
                groups[g2[0]][g2[1]] = (g2[0], g2[1], groups[g1[0]][g1[1]][2] + groups[g2[0]][g2[1]][2])


        value2col = [dict() for _ in range(N)]
        value2row = [dict() for _ in range(M)]
        for i in range(M):
            for j in range(N):
                v = matrix[i][j]
                if v not in value2col[j]:
                    value2col[j][v] = [(i, j)]
                else: value2col[j][v].append((i, j))
                if v not in value2row[i]:
                    value2row[i][v] = [(i, j)]
                else: value2row[i][v].append((i, j))

        for i in range(M):
            for k, l in value2row[i].items():
                first = l[0]
                for index in range(1, len(l)):
                    union(first[0], first[1], l[index][0], l[index][1])

        for j in range(N):
            for k, l in value2col[j].items():
                first = l[0]
                for index in range(1, len(l)):
                    union(first[0], first[1], l[index][0], l[index][1])

        groups2indices = dict()
        for i in range(M):
            for j in range(N):
                g = find(i, j)
                if g not in groups2indices:
                    groups2indices[g] = [(i, j)]
                else:
                    groups2indices[g].append((i, j))

        heap = []
        for g, l in groups2indices.items():
            heap.append((matrix[l[0][0]][l[0][1]], g))
        heapq.heapify(heap)

        row_level = [0 for _ in range(M)]
        col_level = [0 for _ in range(N)]

        while heap:
            v, g = heapq.heappop(heap)
            level = 1
            for i, j in groups2indices[g]:
                level = max(level, max(row_level[i], col_level[j]) + 1)

            for i, j in groups2indices[g]:
                row_level[i] = level
                col_level[j] = level
                answer[i][j] = level
        return answer

    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        products.sort()

        trie = dict()
        t = trie
        END = "END"
        for c in searchWord:
            t[c] = dict()
            t[END] = []
            t = t[c]

        for i, p in enumerate(products):
            t = trie
            for c in p:
                if c not in t: break
                if len(t[END]) != 3:
                    t[END].append(i)
                t = t[c]

        t = trie
        result = []
        for i, c in enumerate(searchWord):
            r = []
            for index in t[END]:
                r.append(products[index])
            result.append(r)
            t = t[c]
        return result

    def isMatch(self, s: str, p: str) -> bool:
        tokens = []
        i = 0
        while i < len(p):
            if i + 1 < len(p) and p[i + 1] == '*':
                tokens.append(p[i:i + 2])
                i += 2
            else:
                tokens.append(p[i])
                i += 1
        dp = [[False for _ in range(len(s) + 1)] for _ in range(len(tokens))]

        for i in range(len(tokens)):
            for j in range(len(s) + 1):
                if len(tokens[i]) == 2: # *
                    if j == 0:
                        if i == 0:
                            is_matched = True
                        else: is_matched = dp[i - 1][j]
                    else:
                        if i == 0:
                            is_matched = (j == 1 and (s[j - 1] == tokens[i][0] or tokens[i][0] == '.')) or \
                                         ((s[j - 1] == tokens[i][0] or tokens[i][0] == '.') and dp[i][j - 1])
                        else:
                            is_matched = dp[i - 1][j] or (dp[i][j - 1] and (s[j - 1] == tokens[i][0] or tokens[i][0] == '.'))
                elif tokens[i] == '.':
                    if j == 0:
                        is_matched = False
                    else:
                        if i == 0:
                            is_matched = j == 1
                        else:
                            is_matched = dp[i - 1][j - 1]
                else:
                    if j == 0:
                        is_matched = False
                    else:
                        if i == 0:
                            is_matched = (j == 1 and s[j - 1] == tokens[i][0])
                        else:
                            is_matched = (dp[i - 1][j - 1] and s[j - 1] == tokens[i][0])
                dp[i][j] = is_matched
        return dp[-1][-1]

    def reverse(self, x: int) -> int:
        is_negative = x < 0
        x = abs(x)
        x_str = str(x)

        reversed_str = []
        leading_zero = True
        for i in range(len(x_str)):
            if leading_zero and x_str[i] == '0':
                i += 1
            else:
                leading_zero = False
                reversed_str.append(x_str[i])
        reversed_x_str = ''.join(reversed(reversed_str))


        max_negative_str = "2147483648"
        max_positive_str = "2147483647"


        def isSGreaterThanBase(s, base):
            if len(base) > len(s): return False
            if len(base) < len(s): return True
            return s > base

        if not reversed_x_str: return 0

        if is_negative:
            if isSGreaterThanBase(reversed_x_str, max_negative_str):
                return 0
            else: return -int(reversed_x_str)
        else:
            if isSGreaterThanBase(reversed_x_str, max_positive_str):
                return 0
            else: return int(reversed_x_str)

    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        result = []
        left = lower
        for n in nums:
            if n == left:
                left = n + 1
            else:
                result.append([left, n - 1])
                left = n + 1
        if left <= upper:
            result.append([left, upper])
        return result

    def lengthOfLongestSubstring(self, s: str) -> int:
        from collections import Counter
        counter = Counter()
        left = 0
        max_length = 0
        for i, c in enumerate(s):
            counter[c] += 1
            while counter[c] > 1:
                counter[s[left]] -= 1
                left += 1
            max_length = max(max_length, (i - left) + 1)
        return max_length

    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        if head is None:
            new_node = Node(insertVal)
            new_node.next = new_node
            return new_node

        node = head
        def insert_behind(node):
            new_node = Node(insertVal)
            new_node.next = node.next
            node.next = new_node
            return head


        seen = set()

        while True:
            seen.add(node)
            if node.val <= insertVal <= node.next.val:
                return insert_behind(node)
            elif node.next == node:
                return insert_behind(node)
            elif node.val < node.next.val:
                node = node.next
            elif node.val > node.next.val:
                if insertVal >= node.val:
                    return insert_behind(node)
                elif insertVal <= node.next.val:
                    return insert_behind(node)
                else:
                    node = node.next
            elif node.val == node.next.val:
                if node in seen and node.next in seen:
                    return insert_behind(node)
                if insertVal > node.val:
                    node = node.next
                else:
                    node = node.next

    def jump(self, nums: List[int]) -> int:
        import math
        dp = [math.inf] * len(nums)

        dp[-1] = 0
        for i in range(len(nums) - 2, -1, -1):
            for j in range(nums[i] + 1):
                if i + j >= len(nums): break
                dp[i] = min(dp[i], 1 + dp[i + j])
        return dp[0]

    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        points = dict()
        for i, s in enumerate(schedule):
            for inter in s:
                if inter.start not in points:
                    points[inter.start] = {
                        '+': set(),
                        '-': set()
                    }
                if inter.end not in points:
                    points[inter.end] = {
                        '+': set(),
                        '-': set()
                    }
                points[inter.start]['+'].add(i)
                points[inter.end]['-'].add(i)

        left = min(points.keys())
        result = []
        running_set = set()
        for p in sorted(points.keys()):
            prev_set_size = len(running_set)
            d = points[p]
            for i in d['+']:
                running_set.add(i)
            for i in d['-']:
                running_set.remove(i)

            if prev_set_size == 0 and left != p:
                result.append(Interval(left, p))
            left = p
        return result

    def minimumMountainRemovals(self, nums: List[int]) -> int:
        def get_dp(array):
            dp = [i for i in range(len(array))]
            for i in range(1, len(array)):
                for j in range(i - 1, -1, -1):
                    if array[j] >= array[i]: continue
                    dp[i] = min(dp[i], i - j - 1 + dp[j])
            return dp

        front_dp = get_dp(nums)
        back_dp = list(reversed(get_dp(list(reversed(nums)))))
        minimum = len(nums) - 3

        for i in range(1, len(nums) - 1):
            if front_dp[i] == i or back_dp[i] == len(nums) - i - 1:
                # you can't delete all numbers
                continue
            minimum = min(minimum, front_dp[i] + back_dp[i])
        return minimum

    def lengthOfLIS(self, nums: List[int]) -> int:
        seq = [nums[0]]

        def insert_at(v):
            left, right = 0, len(seq) - 1
            while left <= right:
                mid = (left + right) // 2
                if seq[mid] >= v:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        for i in range(1, len(nums)):
            n = nums[i]
            index = insert_at(n)
            if index == len(seq):
                seq.append(n)
            else:
                seq[index] = n
        return len(seq)

    def multiply(self, num1: str, num2: str) -> str:
        from collections import deque
        if num1 == "0" or num2 == "0": return "0"

        result = [deque() for _ in range(len(num1))]
        for i in range(len(num1) - 1, -1, -1):
            residue = 0
            for j in range(len(num2) - 1, -1, -1):
                multi = int(num1[i]) * int(num2[j])
                multi += residue
                result[i].appendleft(multi % 10)
                residue = multi // 10
            if residue != 0:
                result[i].appendleft(residue)
            for j in range(len(num1) - i - 1):
                result[i].append(0)

        def sum_dequeues(arr1, arr2):
            r = deque()
            residue = 0
            while arr1 or arr2:
                if not arr1:
                    left = 0
                else:
                    left = arr1[-1]
                    arr1.pop()
                if not arr2:
                    right = 0
                else:
                    right = arr2[-1]
                    arr2.pop()
                multi = left + right + residue
                r.appendleft(multi % 10)
                residue = multi // 10
            if residue != 0:
                r.appendleft(residue)
            return r

        last_queue = result[0]
        for i in range(1, len(result)):
            last_queue = sum_dequeues(last_queue, result[i])
        return ''.join([str(x) for x in last_queue])

    def search(self, nums: List[int], target: int) -> int:
        def find_pivot_point(nums):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[left] > nums[mid] and nums[mid] <= nums[right]:
                    right = mid - 1
                elif nums[left] <= nums[right]:
                    if nums[left] < nums[0]:
                        return left
                    else:
                        return right + 1
                else:
                    left = mid + 1

        def search(nums, left, right, target):
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target: return mid
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return -1

        if nums[0] <= nums[-1]:
            return search(nums, 0, len(nums) - 1, target)
        pivot = find_pivot_point(nums)
        index = search(nums, 0, pivot - 1, target)
        if index == -1:
            return search(nums, pivot, len(nums) - 1, target)
        return index

    def countSmaller(self, nums: List[int]) -> List[int]:
        import math
        class SegmentTree:
            def __init__(self, N):
                self.arr = [0] * (2 * N)
                self.N = N

            def increment(self, index):
                index = index + self.N
                self.arr[index] += 1
                index = index >> 1

                while index > 0:
                    self.arr[index] += 1
                    index = index >> 1

            def query(self, left, right):
                l, r = left + self.N, right + self.N
                result = 0
                while l < r:
                    if l % 2 == 1:
                        result += self.arr[l]
                        l += 1
                    if r % 2 == 1:
                        r -= 1
                        result += self.arr[r]
                    l = l >> 1
                    r = r >> 1
                return result

        nToIndex = dict()
        nToIndex[-math.inf] = 0

        for n in sorted(nums):
            if n not in nToIndex:
                nToIndex[n] = len(nToIndex)

        tree = SegmentTree(len(nToIndex))

        result = [0] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            n = nums[i]
            result[i] = tree.query(0, nToIndex[n])
            tree.increment(nToIndex[n])
        return result


    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        init = [[], []]
        def update(array, id):
            for inter in array:
                if id == 'A':
                    init[0].append((inter[0], id, 0))
                    init[0].append((inter[1], id, 1))
                else:
                    init[1].append((inter[0], id, 0))
                    init[1].append((inter[1], id, 1))
        update(firstList, 'A')
        update(secondList, 'B')
        intervals = []
        i, j = 0, 0
        while i < len(init[0]) or j < len(init[1]):
            if i < len(init[0]) and j < len(init[1]):
                if init[0][i][0] <= init[1][j][0]:
                    intervals.append(init[0][i])
                    i += 1
                else:
                    intervals.append(init[1][j])
                    j += 1
            elif i < len(init[0]):
                intervals.append(init[0][i])
                i += 1
            else:
                intervals.append(init[1][j])
                j += 1

        s = set()
        i = 0
        left = None
        result = []
        while i < len(intervals):
            pit, identifier, op = intervals[i]
            actions = set()
            if op == 0:
                s.add(identifier)
            else:
                s.remove(identifier)
            actions.add(identifier)

            while i + 1 < len(intervals) and intervals[i + 1][0] == pit:
                _, identifier, op = intervals[i + 1]
                if op == 0:
                    s.add(identifier)
                else:
                    s.remove(identifier)
                actions.add(identifier)
                i = i + 1

            if left is not None:
                if len(s) < 2:
                    result.append([left, pit])
                left = None
            else:
                if len(s) == 2:
                    left = pit
                elif len(actions) == 2:
                    result.append([pit, pit])
            i += 1
        return result

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        count = 0
        node = head
        while node is not None:
            count += 1
            node = node.next

        target = count - n - 1
        node = head
        prev = None
        while target >= 0:
            prev = node
            node = node.next
            target -= 1
        if prev is None:
            return head.next
        prev.next = node.next
        return head

    def divide(self, dividend: int, divisor: int) -> int:
        max_negative = -2147483648
        max_positive = 2147483647
        if dividend == 0: return 0
        sign = False
        if dividend < 0 < divisor: sign = True
        elif divisor < 0 < dividend: sign = True

        dividend = abs(dividend)
        divisor = abs(divisor)

        def divide(dividend, divisor):
            if divisor > dividend: return 0
            if divisor == dividend: return 1

            i = 1
            r = divisor
            while True:
                if r + r > dividend: break
                r += r
                i += i
            return i + divide(dividend - r, divisor)

        if sign:
            r = -divide(dividend, divisor)
        else: r = divide(dividend, divisor)
        if r <= max_negative: return max_negative
        if r >= max_positive: return max_positive

    def validIPAddress(self, queryIP: str) -> str:
        valid_ipv4_characters = set()
        for i in range(10):
            valid_ipv4_characters.add(str(i))
        for c in ['a', 'b', 'c', 'd', 'e', 'f',
                  'A', 'B', 'C', 'D', 'E', 'F']:
            valid_ipv4_characters.add(c)

        if '.' in queryIP:
            subs = queryIP.split('.')
            if len(subs) != 4: return "Neither"
            for i in range(len(subs)):
                try:
                    s = int(subs[i])
                    if s < 0 or s > 255:
                        return "Neither"
                    if len(subs[i]) > 1 and subs[i][0] == '0':
                        return "Neither"
                except:
                    return "Neither"
            return "IPv4"
        elif ':' in queryIP:
            subs = queryIP.split(':')
            if len(subs) != 8: return "Neither"
            for s in subs:
                if len(s) > 4 or len(s) == 0: return "Neither"
                for c in s:
                    if c not in valid_ipv4_characters:
                        return "Neither"
            return "IPv6"
        else:
            return "Neither"

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:

        for i in range(len(flowerbed)):
            if flowerbed[i] == 1: continue
            can_plant = True
            if i >= 1:
                can_plant = can_plant and flowerbed[i - 1] == 0
            if i < len(flowerbed) - 1:
                can_plant = can_plant and flowerbed[i + 1] == 0
            if can_plant:
                flowerbed[i] = 1
                n -= 1
            if n <= 0: break
        return n <= 0

    def maxEvents(self, events: List[List[int]]) -> int:
        import heapq
        events = [(e[0], e[1]) for e in events]
        events.sort()
        count = 0
        heap = []


        i = 0

        start = events[0][0]
        available = start
        while i < len(events):
            available = start
            while i < len(events) and events[i][0] == available:
                heapq.heappush(heap, events[i][1])
                i += 1
            if i >= len(events): break
            start = events[i][0]
            while available < start:
                while heap and heap[0] < available:
                    heapq.heappop(heap)
                if not heap: break
                heapq.heappop(heap)
                available += 1
                count += 1

        while True:
            while heap and heap[0] < available:
                heapq.heappop(heap)
            if not heap: break
            heapq.heappop(heap)
            count += 1
            available += 1
        return count

    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        # we rob the first one
        dp = [0] * len(nums)
        for i in range(len(nums) - 2, 0, -1):
            if i + 2 < len(nums):
                dp[i] = max(dp[i], nums[i] + dp[i + 2])
            dp[i] = max(dp[i], dp[i + 1], nums[i])
        if 2 < len(nums):
            dp[0] = max(dp[0], nums[0] + dp[2])
        else: dp[0] = nums[0]


        # we don't rob the first one
        dp2 = [0] * len(nums)
        for i in range(len(nums) - 1, 0, -1):
            if i + 2 < len(nums):
                dp2[i] = max(dp2[i], nums[i] + dp2[i + 2])
            if i + 1 < len(nums):
                dp2[i] = max(dp2[i], dp2[i + 1], nums[i])
            else:
                dp2[i] = max(dp2[i], nums[i])
        return max(max(dp), max(dp2))

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def find_first(target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def find_last(target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            return right
        r = [find_first(target), find_last(target)]
        if r[0] >= len(nums) or r[0] < 0 or nums[r[0]] != target:
            r[0] = -1
        if r[1] >= len(nums) or r[0] < 0 or nums[r[1]] != target:
            r[1] = -1
        return r

    def maxScore(self, s: str) -> int:
        dp = [0]
        for i, c in enumerate(s):
            r = 0
            if c == '0':
                r = 1
            if i > 0:
                dp[i] = r + dp[i - 1]
            else: dp[i] = r

        result = 0
        running_sum = 0
        for i in range(len(s) - 1, 0, -1):
            r = 0
            if s[i] == '1':
                r = 1
            running_sum += r
            result = max(result, running_sum + dp[i - 1])
        return result

    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        M = len(grid)
        N = len(grid[0])
        start_x, start_y = None, None
        end_x, end_y = None, None
        count = 0
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    start_x = i
                    start_y = j
                elif grid[i][j] == 2:
                    end_x = i
                    end_y = j
                if grid[i][j] != -1:
                    count += 1

        def gen(i, j):
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if i + dx < 0 or i + dx >= M: continue
                if j + dy < 0 or j + dy >= N: continue
                if grid[i + dx][j + dy] == -1: continue
                yield i + dx, j + dy

        def get_digit(i, j):
            return 1 << (i * N + j)

        dp = dict()
        def recur(i, j, mask, total):
            nonlocal count
            if (i, j, mask) in dp: return dp[i, j, mask]
            total += 1
            if i == end_x and j == end_y:
                if total == count:
                    return 1
                else: return 0

            mask = mask | (get_digit(i, j))

            result = 0
            for ni, nj in gen(i, j):
                digit = get_digit(ni, nj)
                if (mask & digit) != 0: continue
                result += recur(ni, nj, mask, total)

            dp[i, j, mask] = result
            return result
        r = recur(start_x, start_y, 0, 0)
        return r

    def findMaximumScore(self, nums: List[int]) -> int:
        stack = []
        for i in range(len(nums) - 2, -1, -1):
            while stack and stack[-1][0] <= nums[i]:
                stack.pop(-1)
            stack.append((nums[i], i))

        score = 0
        while stack:
            n, i = stack.pop(-1)
            if stack:
                prevIndex = stack[-1][1]
            else:
                prevIndex = len(nums) - 1
            score += n * (prevIndex - i)
        return score

    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        M = len(mat)
        N = len(mat[0])
        import math
        from collections import deque
        result = [[math.inf for _ in range(N)] for _ in range(M)]

        def gen(i, j):
            for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if i + di < 0 or i + di >= M: continue
                if j + dj < 0 or j + dj >= N: continue
                yield i + di, j + dj

        queue = deque()
        for i in range(M):
            for j in range(N):
                if mat[i][j] != 0: continue
                result[i][j] = 0
                for ni, nj in gen(i, j):
                    if mat[ni][nj] == 0: continue
                    queue.append((1, ni, nj))

        while queue:
            d, i, j = queue.popleft()
            if d >= result[i][j]: continue
            result[i][j] = d
            for ni, nj in gen(i, j):
                if mat[ni][nj] == 0: continue
                if 1 + d >= result[ni][nj]: continue
                queue.append((1 + d, ni, nj))
        return result

    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        dp = [0] * len(nums)
        minIndex = dict()
        maxIndex = dict()

        for i in range(len(nums)):
            if nums[i] not in minIndex:
                minIndex[nums[i]] = i
            else: minIndex[nums[i]] = min(i, minIndex[nums[i]])
            if nums[i] not in maxIndex:
                maxIndex[nums[i]] = i
            else: maxIndex[nums[i]] = max(i, maxIndex[nums[i]])

        for i in range(len(nums)):
            min_index = minIndex[nums[i]]
            max_index = maxIndex[nums[i]]
            if max_index > i: continue
            j = i
            while j >= 0:
                while True:
                    min_index = min(min_index, minIndex[nums[j]])
                    max_index = max(max_index, maxIndex[nums[j]])
                    if max_index > i: break
                    if min_index == j: break
                    j -= 1
                if max_index > i: break
                if j > 0:
                    dp[i] = (dp[i] + dp[j - 1]) % mod
                else:
                    dp[i] += 1
                j -= 1
        return dp[-1]

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        queue = deque()
        result = []

        i = 0
        while i < k:
            while queue:
                if queue[-1][0] <= nums[i]:
                    queue.pop()
                else: break
            queue.append((nums[i], i))
            i += 1
        result.append(queue[0][0])
        while i < len(nums):
            while queue:
                if queue[-1][0] <= nums[i]:
                    queue.pop()
                else: break
            queue.append((nums[i], i))
            while queue and queue[0][1] <= i - k:
                queue.popleft()
            result.append(queue[0][0])
            i += 1
        return result

    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        cars = [(position[i], speed[i]) for i in range(len(position))]
        cars.sort()
        maxTime = -1
        result = 0

        for car in reversed(cars):
            pos, s = car[0], car[1]
            t = (target - pos) / s
            if t > maxTime:
                result += 1
                maxTime = t
        return result

    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        result = [0] * len(heights)
        # decreasing
        stack = []

        for i in range(len(heights) - 1, -1, -1):
            while stack and stack[-1] <= heights[i]:
                stack.pop(-1)
                result[i] += 1
            if stack: result[i] += 1
            stack.append(heights[i])
        return result

    def minDistance(self, houses: List[int], k: int) -> int:
        houses.sort()
        import math
        def get_cost(ranges: List[int]):
            if len(ranges) % 2 == 1:
                median = ranges[len(ranges) // 2]
            else:
                median = (ranges[len(ranges) // 2] + ranges[len(ranges) // 2 - 1]) / 2.0
            cost = 0
            for r in ranges:
                cost += abs(r - median)
            return cost;

        dp = [[math.inf for _ in range(k + 1)] for _ in range(len(houses))]
        for i in range(len(houses)):
            for sub_k in range(1, k + 1):
                if sub_k >= i + 1:
                    dp[i][sub_k] = 0
                    continue
                ranges = []
                for j in range(i, -1, -1):
                    ranges.append(houses[j])
                    cost = get_cost(ranges)
                    if j > 0:
                        dp[i][sub_k] = min(dp[i][sub_k], cost + dp[j - 1][sub_k - 1])
                    else:
                        dp[i][sub_k] = min(dp[i][sub_k], cost)
        return int(dp[-1][-1])

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1: return head

        def reverse(head):
            if head is None or head.next is None: return head
            # 1 -> 2 -> 3 -> 4 -> 5
            # 2 -> 1 -> 3 -> 4 -> 5
            # 3 -> 2 -> 1 -> 4 -> 5
            tail = head
            node = head.next
            while node is not None:
                next_node = node.next
                tail.next = next_node
                node.next = head
                head = node
                node = next_node
            tail.next = None
            return head, tail

        head = head
        current_head = head
        current_tail = None
        count = 0

        node = head
        head_parent = None
        while node is not None:
            count += 1
            current_tail = node
            node = node.next
            if count % k == 0:
                current_tail.next = None
                new_head, new_tail = reverse(current_head)
                new_tail.next = node
                current_head = node
                if count == k:
                    head = new_head
                if head_parent is not None:
                    head_parent.next = new_head
                head_parent = new_tail
                current_tail = None
        return head

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        counter = Counter(words)
        length = len(words[0])

        def find(start):
            result = []
            nonlocal length
            matches = []
            for i in range(start, len(s), length):
                j = i + length
                w = s[i:j]
                if w in counter:
                    matches.append(w)
                else:
                    matches.append(None)
            new_counter = Counter()
            left = 0
            right = 0
            while right < len(matches):
                if matches[right] == None:
                    new_counter = Counter()
                    left = right + 1
                    right += 1
                    continue

                new_counter[matches[right]] += 1
                while new_counter[matches[right]] > counter[matches[right]]:
                    new_counter[matches[left]] -= 1
                    left += 1

                if right - left + 1 == len(words):
                    result.append(start + left * length)
                right += 1
            return result
        final_result = []
        for i in range(length):
            final_result.extend(find(i))
        return final_result

    def minStickers(self, stickers: List[str], target: str) -> int:
        import math
        def get_index(c):
            return ord(c) - ord('a')

        counts = [[0] * 26 for _ in range(len(stickers))]
        for i, s in enumerate(stickers):
            for c in s:
                counts[i][get_index(c)] += 1

        dp = dict()
        def get_new_mask(current_mask, char_count, multiplier):
            new_count = [c * multiplier for c in char_count]
            new_mask = current_mask
            for i in range(len(target)):
                if (1 << i) & current_mask == 0:
                    if new_count[get_index(target[i])] > 0:
                        new_count[get_index(target[i])] -= 1
                        new_mask = new_mask | (1 << i)
            return new_mask

        def get_max_multiplier(current_mask, char_count):
            max_mult = 0
            new_count = [0] * 26
            for i in range(len(target)):
                if (1 << i) & current_mask == 0:
                    new_count[get_index(target[i])] += 1
            for i in range(26):
                if char_count[i] != 0:
                    max_mult = max(max_mult, new_count[i] / char_count[i])
            return math.ceil(max_mult)

        def can_fit(current_mask, char_count):
            for i in range(len(target)):
                if (1 << i) & current_mask == 0:
                    if char_count[get_index(target[i])] == 0: return False
            return True

        def recur(i, mask):
            if (i, mask) in dp: return dp[i, mask]
            max_mult = get_max_multiplier(mask, counts[i])

            if can_fit(mask, counts[i]):
                min_result = max_mult
            else:
                min_result = math.inf
            if i < len(counts) - 1:
                for j in range(max_mult + 1):
                    min_result = min(min_result, j +
                                     recur(i + 1, get_new_mask(mask, counts[i], j)))
            dp[i, mask] = min_result
            return min_result
        r = recur(0, 0)
        if r == math.inf:
            return -1
        return r

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        words = set(wordDict)
        dp = [[] for _ in range(len(s))]
        for j in range(len(s)):
            for i in range(j, -1, -1):
                n_w = s[i:j + 1]
                if n_w not in words: continue
                if i == 0:
                    dp[j].append([n_w])
                else:
                    for arr in dp[i - 1]:
                        new_arr = [w for w in arr]
                        new_arr.append(n_w)
                        dp[j].append(new_arr)
        return [" ".join(arr) for arr in dp[-1]]


    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if root is None: return 0
        import math
        from collections import Counter
        max_path_sum = -math.inf
        counter = Counter()
        max_sum = dict()
        stack = [root]

        while stack:
            node = stack.pop(-1)
            counter[node] += 1
            if counter[node] == 1:
                stack.append(node)
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
            else:
                if node.left is None:
                    left_value = 0
                else: left_value = max_sum[node.left]
                if node.right is None:
                    right_value = 0
                else: right_value = max_sum[node.right]
                max_path_sum = max(max_path_sum, node.val,
                                   left_value + right_value + node.val,
                                   node.val + left_value,
                                   node.val + right_value)
                max_sum[node] = max(node.val + max(left_value, right_value), node.val)
        return max_path_sum

    def longestValidParentheses(self, s: str) -> int:
        longest = 0
        stack = []
        for i, c in enumerate(s):
            if c == '(':
                stack.append((c, i))
                continue
            if stack:
                if stack[-1][0] == '(':
                    stack.pop(-1)
                    if stack:
                        longest = max(longest, i - stack[-1][1])
                    else:
                        longest = max(longest, i + 1)
                else:
                    stack.append((c, i))
            else:
                stack.append((c, i))
        return longest

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        M = len(matrix)
        N = len(matrix[0])
        rows_zero = [False] * M
        cols_zero = [False] * N

        for i in range(M):
            for j in range(N):
                if matrix[i][j] == 0:
                    rows_zero[i] = True
                    cols_zero[j] = True

        for i in range(M):
            for j in range(N):
                if rows_zero[i] or cols_zero[j]:
                    matrix[i][j] = 0

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        from collections import Counter
        counter = Counter(nums)
        start = 0
        for k in [0, 1, 2]:
            for _ in range(counter[k]):
                nums[start] = k
                start += 1

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if len(matrix) == 0: return []
        M = len(matrix)
        N = len(matrix[0])

        result = []
        i, j = 0, 0
        col_length = N
        row_length = M

        while len(result) < M * N:

            for k in range(col_length):
                result.append(matrix[i][j + k])
            i, j = i + 1, j + col_length - 1
            row_length -= 1
            if len(result) >= M * N: break

            for k in range(row_length):
                result.append(matrix[i + k][j])
            i, j = i + row_length - 1, j - 1
            col_length -= 1
            if len(result) >= M * N: break

            for k in range(col_length):
                result.append(matrix[i][j - k])
            i, j = i - 1, j - col_length + 1
            row_length -= 1
            if len(result) >= M * N: break

            for k in range(row_length):
                result.append(matrix[i - k][j])
            i, j = i - row_length + 1, j + 1
            col_length -= 1
            if len(result) >= M * N: break
        return result

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        for inter in intervals:
            if newInterval is not None:
                if newInterval[0] > inter[-1]:
                    result.append(inter)
                elif newInterval[1] < inter[0]:
                    result.append(newInterval)
                    result.append(inter)
                    newInterval = None
                else:
                    result.append([min(newInterval[0], inter[0]), max(newInterval[1], inter[1])])
                    newInterval = None
            else:
                if not result:
                    result.append(inter)
                elif result[-1][-1] < inter[0]:
                    result.append(inter)
                else:
                    result[-1][-1] = max(result[-1][-1], inter[1])
        if newInterval is not None:
            result.append(newInterval)
        return result


    def check(self, nums: List[int]) -> bool:
        smallest = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] > nums[i + 1]: break
            if nums[i] <= nums[smallest]: smallest = i


        for i in range(len(nums) - 1):
            if nums[(smallest + i) % len(nums)] > nums[(smallest + i + 1) % len(nums)]: return False

        return True















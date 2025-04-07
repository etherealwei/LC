from cgitb import reset
from email.feedparser import headerRE
from functools import total_ordering, lru_cache
from itertools import count


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

from typing import *

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        import heapq
        prev = 1
        heap = [2, 3, 5]
        for i in range(1, n):
            while heap[0] <= prev:
                heapq.heappop(heap)
            heapq.heappush(heap, prev * 2)
            heapq.heappush(heap, prev * 3)
            heapq.heappush(heap, prev * 5)
            prev = heap[0]
        return prev

    def maximumValueSum(self, board: List[List[int]]) -> int:
        import heapq
        value = 0
        count = 3
        heap = []
        n = len(board)
        m = len(board[0])
        seen_row = set()
        seen_col = set()
        for i in range(n):
            for j in range(m):
                heapq.heappush(heap, (-board[i][j], (i, j)))

        while heapq and count > 0:
            v, index = heapq.heappop(heap)
            if index[0] in seen_row: continue
            if index[1] in seen_col: continue
            seen_row.add(index[0])
            seen_col.add(index[1])
            value -= v
            count -= 1

        return value

    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        # to delete the smallest number A: op = 1 + A_index
        # to delete the second-smallest number B:
        # if A is left to B: 4 3 1 6 7 2 -> 1 6 7 2 4 3 -> 6 7 2 4 3 -> 2 4 3 6 7
        # op = B_index - A_index
        # if A is right to B: 4 3 2 1: op = 1 + B_index
        # to delete the third-smallest number C: 4 1 2 3
        import math
        nums = [-math.inf] + nums
        operations = 0
        # need to know for each element, what's the index of the closest smaller element left to it
        index = [-1 for _ in range(len(nums))]
        stack = [0]
        for i in range(1, len(nums)):
            while True:
                if nums[stack[-1]] >= nums[i]:
                    stack.pop(-1)
                else: break
            index[i] = stack[-1]
            stack.append(i)

        for i in range(1, len(nums)):
            operations += i - index[i]
        return operations

    def minAnagramLength(self, s: str) -> int:
        from collections import Counter

        def search(length):
            if length == 0: return False
            original_counter = Counter()
            for i in range(length):
                original_counter[s[i]] += 1
            anagram_counter = original_counter.copy()
            counter = length
            for i in range(length, len(s)):
                anagram_counter[s[i]] -= 1
                if anagram_counter[s[i]] < 0: return False
                counter -= 1
                if counter == 0:
                    counter = length
                    anagram_counter = original_counter.copy()
            if counter != 0 and counter != length: return False
            return True

        for i in range(1, len(s) + 1):
            if len(s) % i == 0 and search(i): return i

    def numSimilarGroups(self, strs: List[str]) -> int:
        strs = list(set(strs))
        groups = [(i, 1) for i in range(len(strs))]
        def find(i):
            if groups[i][0] != i:
                g = find(groups[i][0])
                groups[i] = (g, groups[i][1])
                return g
            else: return i

        def union(i, j):
            group_i = find(i)
            group_j = find(j)
            if group_i == group_j: return
            if groups[group_i][1] > groups[group_j][1]:
                groups[group_j] = (group_i, groups[group_j][1])
                groups[group_i] = (group_i, groups[group_i][1] + groups[group_j][1])
            else:
                groups[group_i] = (group_j, groups[group_i][1])
                groups[group_j] = (group_j, groups[group_i][1] + groups[group_j][1])

        all_numbers = {s : i for i, s in enumerate(strs)}

        for index, s in enumerate(strs):
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    neighbor = s[:i] + s[j] + s[i + 1:j] + s[i] + s[j + 1:]
                    if neighbor not in all_numbers: continue
                    union(index, all_numbers[neighbor])

        for i in range(len(strs)):
            find(i)
        return len(set(g[0] for g in groups))

    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        dp = [-1 for _ in range(len(nums))]
        mod = 10**9 + 7

        def recur(start_index):
            if start_index >= len(nums): return 1
            if dp[start_index] != -1: return dp[start_index]
            total = 0
            size = 1
            while size + start_index <= len(nums):
                seen = set()
                for i in range(start_index, start_index + size):
                    seen.add(nums[i])
                for i in range(start_index + size, len(nums)):
                    if nums[i] in seen:
                        for j in range(start_index + size, i + 1):
                            seen.add(nums[j])
                        size = i - start_index + 1
                total = (total + recur(start_index + size)) % mod
                size += 1
            dp[start_index] = total % mod
            return dp[start_index]
        return recur(0)

    def countSubarrays(self, nums: List[int], k: int) -> int:
        prefix_sum = []
        for n in nums:
            if not prefix_sum:
                prefix_sum.append(n)
            else:
                prefix_sum.append(n + prefix_sum[-1])

        def count(i):
            left = 0
            right = i
            while left <= right:
                mid = (left + right) // 2
                if mid == 0:
                    mid_prefix_sum = 0
                else: mid_prefix_sum = prefix_sum[mid - 1]
                if (prefix_sum[i] - mid_prefix_sum) * (i - mid + 1) >= k:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        total = 0
        for i in range(len(prefix_sum)):
            index = count(i)
            if index > i: continue
            total += i - index + 1
        return total

    def distinctNames(self, ideas: List[str]) -> int:
        from collections import Counter
        prefix_dict = dict()
        for idea in ideas:
            if idea[0] not in prefix_dict:
                prefix_dict[idea[0]] = {idea[1:]}
            else:
                prefix_dict[idea[0]].add(idea[1:])

        not_found_dict = dict()
        for idea in ideas:
            for k, v in prefix_dict.items():
                x = idea[1:]
                if k == idea[0]: continue
                if x in v: continue
                if idea[0] not in not_found_dict:
                    not_found_dict[idea[0]] = Counter()
                not_found_dict[idea[0]][k] += 1

        total_count = 0
        for idea in ideas:
            for k, v in prefix_dict.items():
                x = idea[1:]
                if k == idea[0]: continue
                if x in v: continue
                if k in not_found_dict:
                    total_count += not_found_dict[k][idea[0]]
        return total_count

    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        robots = [] # tuple of (position, index_in_positions, health, direction)
        for i in range(len(positions)):
            robots.append([positions[i], i, healths[i], directions[i]])
        robots.sort(key=lambda robot: robot[0])
        results = []
        going_left = []
        for robot in reversed(robots):
            if not going_left and robot[-1] == 'R':
                results.append((robot[1], robot[2]))
            elif robot[-1] == 'L':
                going_left.append(robot)
            else:
                while going_left:
                    if going_left[-1][2] > robot[2]:
                        going_left[-1][2] -= 1
                        robot[2] = 0
                        break
                    elif going_left[-1][2] == robot[2]:
                        going_left.pop(-1)
                        robot[2] = 0
                        break
                    else:
                        robot[2] -= 1
                        going_left.pop(-1)
                if robot[2] != 0:
                    results.append((robot[1], robot[2]))
        for gl in going_left:
            results.append((gl[1], gl[2]))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_number = max(k, max(nums))
        highest = 1
        while (1 << highest) <= max_number:
            highest += 1

        # 0010
        # 0000
        # 1010
        # 0010
        # 0101
        closest_zero = [[-1 for _ in range(highest)] for _ in range(len(nums))]
        for i, n in enumerate(nums):
            for j in range(highest):
                if ((1 << j) & n) == 0:
                    closest_zero[i][j] = i
                elif i > 0:
                    closest_zero[i][j] = closest_zero[i - 1][j]

        total = 0
        for i in range(len(nums)):
            left_bound = 0
            right_bound = i
            for j in range(highest):
                if ((1 << j) & k) == 0:
                    right_bound = min(right_bound, closest_zero[i][j])
                else:
                    left_bound = max(left_bound, closest_zero[i][j] + 1)
            if left_bound <= right_bound:
                total += right_bound - left_bound + 1
        return total

    def countKConstraintSubstrings(self, s: str, k: int, queries: List[List[int]]) -> List[int]:
        if k == 0 or not s: return [0] * len(queries)
        # max_left_ending_at_i for each index i in s
        # e.g. [0, 0, 0, 0, 0]
        max_left = [0]
        window_left = 0
        number_of_zero = 0
        number_of_one = 0
        if s[0] == '0':
            number_of_zero += 1
        else: number_of_one += 1

        for i in range(1, len(s)):
            if s[i] == '0':
                number_of_zero += 1
            else:
                number_of_one += 1
            if number_of_zero <= k or number_of_one <= k:
                max_left.append(window_left)
            else:
                while number_of_one > k and number_of_zero > k:
                    if s[window_left] == '0':
                        number_of_zero -= 1
                    else: number_of_one -= 1
                    window_left += 1
                max_left.append(window_left)

        result = []
        for q in queries:
            total = 0
            l, r = q[0], q[1]
            for i in range(l, r + 1):
                total += (i - max(max_left[i], l) + 1)
            result.append(total)
        return result

    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # 3 3 3 -1 3 3 3 -> 3 + 4 or 3 + 3 + 1
        # 3 3 3 -1 3 3 3 3 3 3 -> 3 + 1 + 3
        # 3 3 3 -1 3 -> 3 + 4
        # 3 -1 3 -> 3 + 4
        # -1 3 -1 -> 1 + 4
        # -1 3 3 3 -1 -> 1 + 3 + 1

        decrease_steps = 0
        increase_steps = 0
        total = 0
        for i in range(len(nums)):
            diff = nums[i] - target[i]
            if diff > 0:
                total += max(0, diff - increase_steps)
                increase_steps = diff
                decrease_steps = 0
            elif diff < 0:
                total += max(0, abs(diff) - decrease_steps)
                decrease_steps = abs(diff)
                increase_steps = 0
            else:
                decrease_steps = 0
                increase_steps = 0
        return total

    def numberOfSubstrings(self, s: str, k: int) -> int:
        import math
        counts = [0 for _ in range(26)]

        def check():
            for c in counts:
                if c >= k: return True
            return False

        i = 0
        while i < len(s):
            if not check():
                counts[ord(s[i]) - ord('a')] += 1
                i += 1
            else: break
        minimum_left_index = 0
        while minimum_left_index <= i:
            counts[ord(s[minimum_left_index]) - ord('a')] -= 1
            if not check():
                counts[ord(s[minimum_left_index]) - ord('a')] += 1
                break
            minimum_left_index += 1
        if i == len(s) and not check(): return 0
        total = minimum_left_index + 1
        while i < len(s):
            counts[ord(s[i]) - ord('a')] += 1
            while minimum_left_index <= i:
                counts[ord(s[minimum_left_index]) - ord('a')] -= 1
                if not check():
                    counts[ord(s[minimum_left_index]) - ord('a')] += 1
                    break
                minimum_left_index += 1
            i += 1
            total += minimum_left_index + 1
        return total

    def maxFrequency(self, nums: List[int], k: int) -> int:
        if not nums: return 0
        nums.sort()
        left, right = 1, len(nums)

        def check(desired_freq):
            if desired_freq == 1: return True
            current_ops = 0
            n = nums[0]
            l = 0
            freq = 1
            for i in range(1, len(nums)):
                while current_ops + (i - l) * (nums[i] - n) > k:
                    current_ops -= n - nums[l]
                    l += 1
                    freq -= 1
                freq += 1
                current_ops += (i - l) * (nums[i] - n)
                n = nums[i]
                if freq >= desired_freq: return True
            return False

        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    def minimumSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        min_area = m * n
        for i in range(m):
            min_area = min(min_area,
                           self.minimum_sum_one_rectangle(grid[:i + 1][:]) +
                           self.minimumSumTwoRectangles(grid[i + 1:][:]))
            min_area = min(min_area,
                           self.minimumSumTwoRectangles(grid[:i + 1][:]) +
                           self.minimum_sum_one_rectangle(grid[i + 1:][:]))
        for j in range(n):
            min_area = min(min_area,
                           self.minimum_sum_one_rectangle([row[:j + 1] for row in grid]) +
                           self.minimumSumTwoRectangles([row[j + 1:] for row in grid]))
            min_area = min(min_area,
                           self.minimumSumTwoRectangles([row[:j + 1] for row in grid]) +
                           self.minimum_sum_one_rectangle([row[j + 1:] for row in grid]))
        return min_area


    def minimumSumTwoRectangles(self, grid):
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
        if n == 0: return 0
        min_area = m * n
        for i in range(m):
            min_area = min(min_area,
                           self.minimum_sum_one_rectangle(grid[:i + 1][:]) +
                           self.minimum_sum_one_rectangle(grid[i + 1:][:]))
        for j in range(n):
            min_area = min(min_area,
                           self.minimum_sum_one_rectangle([row[:j + 1] for row in grid]) +
                           self.minimum_sum_one_rectangle([row[j + 1:] for row in grid]))
        return min_area

    def minimum_sum_one_rectangle(self, sub_grid):
        m = len(sub_grid)
        if m == 0: return 0
        n = len(sub_grid[0])
        if n == 0: return 0
        min_m, min_n, max_m, max_n = m + 1, n + 1, -1, -1
        for i in range(m):
            for j in range(n):
                if sub_grid[i][j] == 1:
                    min_m = min(i, min_m)
                    min_n = min(j, min_n)
                    max_m = max(i, max_m)
                    max_n = max(j, max_n)
        if max_m != -1:
            return (max_n - min_n + 1) * (max_m - min_m + 1)
        return 0

    def countOfPairs1(self, nums: List[int]) -> int:
        if not nums: return 0
        # 2, 3, 2 -> [[0, 1], [0, 1, 2], [0, 1]]
        # DP[i][n][m] means how many subarrays there are that satisify arr1[i] + arr2[i] == nums[i] and arr1[i + 1] >= arr1[i] = n and arr2[i + 1] <= arr2[i] = m
        max_nums = max(nums)
        dp = [[0 for _ in range(max_nums + 1)] for _ in range(max_nums + 1)]

        for i in range(len(nums) - 1, -1, -1):
            new_dp = [[0 for _ in range(max_nums + 1)] for _ in range(max_nums + 1)]
            for j in range(nums[i], -1, -1):
                if i == len(nums) - 1:
                    new_dp[j][nums[i] - j] += 1
                    continue
                # nums[i + 1] - sub_j <= nums[i] - j
                for sub_j in range(max(j, nums[i + 1] - nums[i] + j), nums[i + 1] + 1):
                    new_dp[j][nums[i] - j] += dp[sub_j][nums[i + 1] - sub_j]
            dp = new_dp

        total = 0
        for j in range(nums[0] + 1):
            total += dp[j][nums[0] - j]
        return total % (10 ** 9 + 7)

    def countOfPairs(self, nums: List[int]) -> int:
        if not nums: return 0
        max_nums = max(nums)
        dp = [0 for _ in range(max_nums + 1)]

        for i in range(len(nums) - 1, -1, -1):
            new_dp = [0 for _ in range(max_nums + 1)]
            suffix_sum = [0] * (max_nums + 2)
            if i + 1 < len(nums):
                for j in range(nums[i + 1], -1, -1):
                    suffix_sum[j] = dp[j] + suffix_sum[j + 1]

            for j in range(nums[i], -1, -1):
                if i == len(nums) - 1:
                    new_dp[j] += 1
                    continue
                new_dp[j] += suffix_sum[max(j, nums[i + 1] - nums[i] + j)]
            dp = new_dp

        total = 0
        for j in range(nums[0] + 1):
            total += dp[j]
        return total % (10 ** 9 + 7)


    def possibleStringCount(self, word: str, k: int) -> int:
        from math import comb
        # a[2]b[2]c[2]d[2]
        if k > len(word): return 0
        if k == len(word): return 1
        reduced_word = []
        for w in word:
            if not reduced_word:
                reduced_word.append([w, 1])
            elif reduced_word[-1][0] == w:
                reduced_word[-1][1] += 1
            else:
                reduced_word.append([w, 1])

        total = reduced_word[0][1]
        for i in range(1, len(reduced_word)):
            total *= reduced_word[i][1]
        if k <= len(reduced_word): return total

        possibilities = [w[1] - 1 for w in reduced_word if w[1] > 1]
        dp = [0 for _ in range(k)]
        for i in range(len(possibilities) - 1, -1, -1):
            new_dp = [0 for _ in range(k)]
            prefix_sum = []
            if i != len(possibilities) - 1:
                for j in range(k):
                    if not prefix_sum: prefix_sum.append(dp[j])
                    else: prefix_sum.append(prefix_sum[-1] + dp[j])

            for j in range(k - 1, 0, -1):
                if i == len(possibilities) - 1:
                    if j <= possibilities[i]:
                        new_dp[j] = 1
                elif j > possibilities[i]:
                    new_dp[j] = prefix_sum[j] - prefix_sum[j - possibilities[i] - 1]
                else:
                    new_dp[j] = 1 + prefix_sum[j]
            dp = new_dp

        dp.append(0)
        dp[0] = 1
        return (total - sum(dp[:k - len(reduced_word) + 1])) % (10 ** 9 + 7)

    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque
        def build_tree(edges):
            tree = [[] for _ in range(len(edges) + 1)]
            for e in edges:
                tree[e[0]].append(e[1])
                tree[e[1]].append(e[0])
            return tree
        tree1 = build_tree(edges1)
        tree2 = build_tree(edges2)

        def min_diameter(tree):
            if len(tree) == 2: return 1
            # connect to all leaf nodes
            tree.append([i for i in range(len(tree)) if len(tree[i]) == 1])
            for node in tree[-1]:
                tree[node].append(len(tree) - 1)

            degrees = [len(tree[i]) for i in range(len(tree))]
            distance = [-1 for _ in range(len(tree))]
            distance[-1] = 0

            queue = deque()
            queue.append((0, 0))
            while queue:
                node, d = queue.popleft()
                distance[node] = max(distance[node], d)
                for nei in tree[node]:
                    if degrees[nei] <= 0: continue
                    degrees[node] -= 1
                    degrees[nei] -= 1
                    queue.append((nei, d + 1))
            return max(distance) - 1
        return 1 + min_diameter(tree1) + min_diameter(tree2)

    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        seen = [[False for _ in range(n)] for _ in range(m)]

        def get_equals(i, j):
            # 0, 1, 2, 3, 4, 5 size 6
            s = set()
            for pair in [(i, j), (i, n - j - 1), (m - i - 1, j), (m - i - 1, n - j - 1)]:
                s.add(pair)
            return list(s)

        def count_ones(equals):
            return sum(grid[i][j] for i, j in equals)

        total = 0
        equals_map = [[], [], [], [], []]
        for i in range(m):
            for j in range(n):
                if seen[i][j]: continue
                equals = get_equals(i, j)
                for x, y in equals:
                    seen[x][y] = True
                equals_map[len(equals)].append(equals)
        for equals in equals_map[4]:
            number_of_ones = count_ones(equals)
            total += min(number_of_ones, 4 - number_of_ones)

        number_of_zero_and_one = 0
        number_of_two_ones = 0
        number_of_two_zeros = 0 # useless
        for equals in equals_map[2]:
            x1, y1, x2, y2 = equals[0][0], equals[0][1], equals[1][0], equals[1][1]
            if grid[x1][y1] == 1:
                if grid[x2][y2] == 1:
                    number_of_two_ones += 1
                else: number_of_zero_and_one += 1
            else:
                if grid[x2][y2] == 1:
                    number_of_zero_and_one += 1
                else: number_of_two_zeros += 1

        total += number_of_zero_and_one - number_of_zero_and_one % 2
        number_of_two_ones = number_of_two_ones % 2
        number_of_zero_and_one = number_of_zero_and_one % 2

        number_of_only_ones = sum(grid[equals[0][0]][equals[0][1]] for equals in equals_map[1])
        total += number_of_only_ones - number_of_only_ones % 4
        number_of_only_zeros = len(equals_map[0]) - number_of_only_ones
        number_of_only_ones = number_of_only_ones % 4

        if number_of_two_ones == 0:
            if number_of_zero_and_one == 0:
                if number_of_only_zeros >= 4- number_of_only_ones:
                    total += min(number_of_only_ones, 4 - number_of_only_ones)
                else: total += number_of_only_ones
            else:
                if number_of_only_ones == 1:
                    total += 2
                elif number_of_only_ones == 2:
                    total += 1
                elif number_of_only_ones == 3:
                    total += 2
                elif number_of_only_ones == 0:
                    total += 1
        else:
            if number_of_zero_and_one == 0:
                if number_of_only_ones == 1:
                    if number_of_only_zeros >= 1:
                        total += 1
                    else: total += 3
                elif number_of_only_ones == 2:
                    total += 0
                elif number_of_only_ones == 3:
                    total += 1
                elif number_of_only_ones == 0:
                    total += 2
            else:
                total += 1
                if number_of_only_zeros >= 4- number_of_only_ones:
                    total += min(number_of_only_ones, 4 - number_of_only_ones)
                else: total += number_of_only_ones
        return total

    def minimumOperations(self, grid: List[List[int]]) -> int:
        import math
        from collections import Counter
        n = len(grid)
        m = len(grid[0])
        dp = [[math.inf for _ in range(11)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            counter = Counter([grid[j][i] for j in range(n)])
            for k in range(10):
                if i == m - 1:
                    dp[i][k] = n - counter[k]
                else:
                    dp[i][k] = n - counter[k] + min([dp[i + 1][j] for j in range(11) if j != k])
            if i == m - 1:
                dp[i][10] = n
            else:
                dp[i][10] = n + min(dp[i + 1])
        return min(dp[0])

    def maxOperations(self, s: str) -> int:
        consecutive_ones = 0
        total = 0
        consecutive_zero = False
        for i in range(len(s)):
            if s[i] == '1':
                if consecutive_zero:
                    total += consecutive_ones
                consecutive_ones += 1
                consecutive_zero = False
            else:
                consecutive_zero = True
        if consecutive_zero:
            total += consecutive_ones
        return total

    def findMinMoves(self, machines: List[int]) -> int:
        s = sum(machines)
        if s % len(machines) != 0: return -1
        desired_number = s // len(machines)

        previous_flow = 0
        max_moves = 0
        for i in range(len(machines)):
            # if flow > 0, flowing to the left else it's flowing to the right
            current_flow = desired_number - machines[i]
            if previous_flow >= 0:
                if current_flow >= 0:
                    max_moves = max(max_moves, current_flow + previous_flow)
                else:
                    current_flow = previous_flow + current_flow
                    max_moves = max(max_moves, previous_flow + abs(current_flow))
            else:
                if current_flow >= 0:
                    current_flow = current_flow + previous_flow
                    max_moves = max(max_moves, abs(previous_flow) + abs(current_flow))
                else:
                    max_moves = max(max_moves, abs(current_flow + previous_flow))
            previous_flow = current_flow
        return max_moves

    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:

        @lru_cache(None)
        def recur(tx, ty):
            if tx == sx and ty == sy: return True
            if tx < sx or ty < sy: return False

            return recur(tx - ty, ty) or recur(tx, ty - tx)

        return recur(tx, ty)



    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0 for _ in range(n)] for _ in range(k)]

        for i in range(k - 1, -1, -1):
            for j in range(n):
                if i == k - 1:
                    dp[i][j] = max(stayScore[i][j], max(travelScore[j]))
                else:
                    dp[i][j] = max(stayScore[i][j] + dp[i + 1][j], max(
                        travelScore[j][next_j] + dp[i + 1][next_j] for next_j in range(n)))
        return max(dp[0])

    def minimumDifference(self, nums: List[int]) -> int:
        import math
        import heapq
        n = len(nums) // 3
        if n == 1:
            return min(nums[0] - nums[1], nums[0] - nums[2], nums[1] - nums[2])

        min_left_value = [math.inf for _ in range(n)]
        max_right_value = [math.inf for _ in range(n)]

        left_heap = [-nums[i] for i in range(n)]
        heapq.heapify(left_heap)
        original_left_value = sum(nums[:n])
        left_value = original_left_value
        for i in range(n, 2 * n):
            keep = left_value + left_heap[0] + nums[i]
            if keep < left_value:
                heapq.heappop(left_heap)
                heapq.heappush(left_heap, -nums[i])
                left_value = keep
            min_left_value[i - n] = left_value

        right_heap = [nums[i] for i in range(2 * n, len(nums))]
        heapq.heapify(right_heap)
        original_right_value = sum(nums[2 * n:])
        right_value = original_right_value
        for i in range(2 * n - 1, n - 1, -1):
            keep = right_value - right_heap[0] + nums[i]
            if keep > right_value:
                heapq.heappop(right_heap)
                heapq.heappush(right_heap, nums[i])
                right_value = keep
            max_right_value[i - n] = right_value

        minimum = original_left_value - original_right_value
        for i in range(n, 2 * n):
            if i == n:
                minimum = min(minimum, original_left_value - max_right_value[i - n],
                              min_left_value[i - n] - max_right_value[i - n + 1])
            elif i == 2 * n - 1:
                minimum = min(minimum, min_left_value[i - n - 1] - max_right_value[i - n],
                              min_left_value[i - n] - original_right_value)
            else:
                minimum = min(minimum, min_left_value[i - n - 1] - max_right_value[i - n],
                              min_left_value[i - n] - max_right_value[i - n + 1])
        return minimum

    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        # reducing the question to:
        # select walls that satisfy sum(time) > n/2 and minimum cost
        import math
        n = len(cost)
        dp = [[math.inf for _ in range(n + 1)] for _ in range(n)]
        for i in range(min(time[-1], n), 0, -1):
            dp[-1][i] = cost[-1]

        for i in range(n - 2, -1, -1):
            dp[i][0] = 0
            for j in range(1, n + 1):
                for k in range(min(time[i], n) + 1):
                    if k > j: break
                    dp[i][j] = min(dp[i][j], dp[i + 1][j - k] + cost[i], dp[i + 1][j])

        if n % 2 == 1:
            return min(dp[0][n // 2 + 1:])
        else: return min(dp[0][n // 2:])



    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        tree = [[] for _ in range(n)]
        for e in edges:
            tree[e[0]].append((e[1], e[2]))
            tree[e[1]].append((e[0], e[2]))

        def dfs(start, end):
            if start == end: return 0
            seen = [False for _ in range(n)]
            seen[start] = True
            stack = []
            stack.append((start, 0, [0 for _ in range(27)]))

            while stack:
                node, length, counter = stack.pop(-1)
                for nei, weight in tree[node]:
                    if seen[nei]: continue
                    seen[nei] = True
                    new_counter = [c for c in counter]
                    new_counter[weight] += 1
                    if nei == end:
                        return 1 + length - max([(new_counter[i], i) for i in range(len(new_counter))], key=lambda x: x[0])[0]
                    stack.append((nei, length + 1, new_counter))
            return -1

        seen = dict()
        result = []
        for q in queries:
            if (q[0], q[1]) in seen:
                result.append(seen[(q[0], q[1])])
            elif (q[1], q[0]) in seen:
                result.append(seen[(q[1], q[0])])
            else:
                result.append(dfs(q[0], q[1]))
                seen[(q[0], q[1])] = result[-1]
        return result

    def countBalancedPermutations(self, num: str) -> int:
        import math
        from collections import Counter
        if len(num) <= 1: return 0
        nums = [ord(n) - ord('0') for n in num]
        # problem can be reduced to:
        # how many ways to select half of the numbers and make sure
        # they sum to sum(num) / 2
        if sum(nums) % 2 == 1: return 0
        target = sum(nums) // 2
        dp = [[[0 for _ in range(len(nums) // 2 + 1)] for _ in range(max(target, 9) + 1)] for _ in range(len(nums))]

        dp[-1][nums[-1]][1] = 1
        for i in range(len(nums) - 2, -1, -1):
            dp[i][nums[i]][1] = 1
            for t in range(target + 1):
                for j in range(1, len(nums) // 2 + 1):
                    dp[i][t][j] = dp[i][t][j] + dp[i + 1][t - nums[i]][j - 1] + dp[i + 1][t][j]

        mod = 10**9 + 7
        ways = dp[0][target][-1]
        counter = Counter(nums)
        mul = 1
        for k, v in counter.items():
            mul *= math.perm(v, v)
        return (ways
                * math.perm(len(nums) // 2, len(nums) // 2)
                * math.perm(len(nums) - len(nums) // 2, len(nums) - len(nums) // 2)
                // mul) % mod

    def stringSequence(self, target: str) -> List[str]:
        results = []
        prev = ""
        for t in target:
            for i in range(26):
                c = chr(ord('a') + i)
                results.append(prev + c)
                if c == t:
                    prev = prev + c
                    break
        return results

    def validSubstringCount(self, word1: str, word2: str) -> int:
        word2counter = [0 for _ in range(26)]
        for w in word2:
            word2counter[ord(w) - ord('a')] += 1

        counts = []
        for w in word1:
            if not counts:
                counts.append([0 for _ in range(26)])
            else:
                counts.append([x for x in counts[-1]])
            counts[-1][ord(w) - ord('a')] += 1

        def check(left_index, right_index):
            for i in range(26):
                if left_index == 0:
                    if counts[right_index][i] < word2counter[i]: return False
                else:
                    if counts[right_index][i] - counts[left_index - 1][i] < word2counter[i]: return False
            return True

        def find_max_left(right_index):
            left = 0
            right = right_index
            while left <= right:
                mid = (left + right) // 2
                if check(mid, right_index):
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        total = 0
        for i in range(len(word2) - 1, len(word1)):
            left_index = find_max_left(i)
            if left_index > i or left_index < 0: continue
            total += left_index + 1
        return total

    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        # number of zeros for each column should be greater or equal to the number of ones
        # don't bother pick a row with only ones

        N = len(grid)
        M = len(grid[0])
        has_zero = [0] * M
        for i in range(N):
            for j in range(M):
                if grid[i][j] == 0: has_zero[j] = 1
        if sum(has_zero) != M: return []

        for i in range(N):
            if sum(grid[i]) == 0: return [i]
            if sum(grid[i]) == M: continue
            if i == 0: continue
            for j in range(i):
                found = True
                for k in range(M):
                    if grid[i][k] == grid[j][k] == 1:
                        found = False
                        break
                if found: return sorted([i, j])
        return []

    def minTimeToReachI(self, moveTime: List[List[int]]) -> int:
        import math
        import heapq
        N = len(moveTime)
        M = len(moveTime[0])
        min_time = [[math.inf for _ in range(M)] for _ in range(N)]

        def neighbors(i, j):
            for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if N > i + di >= 0 and M > j + dj >= 0:
                    yield i + di, j + dj

        min_time[0][0] = 0
        heap = [(0, 0, 0)]
        while heap:
            time, i, j = heapq.heappop(heap)
            for ni, nj in neighbors(i, j):
                new_time = max(time + 1, moveTime[ni][nj] + 1)
                if min_time[ni][nj] <= new_time: continue
                min_time[ni][nj] = new_time
                heapq.heappush(heap, (new_time, ni, nj))
        return min_time[-1][-1]

    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import math
        import heapq
        N = len(moveTime)
        M = len(moveTime[0])
        min_time = [[[math.inf, math.inf] for _ in range(M)] for _ in range(N)]

        def neighbors(i, j):
            for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if N > i + di >= 0 and M > j + dj >= 0:
                    yield i + di, j + dj

        min_time[0][0] = [0, 0]
        heap = [(0, 0, 0, 0)]
        while heap:
            time, step, i, j = heapq.heappop(heap)
            for ni, nj in neighbors(i, j):
                if step == 0:
                    new_time = max(time + 1, moveTime[ni][nj] + 1)
                    if min_time[ni][nj][0] <= new_time: continue
                    min_time[ni][nj][0] = new_time
                else:
                    new_time = max(time + 2, moveTime[ni][nj] + 2)
                    if min_time[ni][nj][1] <= new_time: continue
                    min_time[ni][nj][1] = new_time
                heapq.heappush(heap, (new_time, (step + 1) % 2, ni, nj))
        return min(min_time[-1][-1])


    def sumOfGoodSubsequences_slow(self, nums: List[int]) -> int:
        # [1,2,2,1]
        # [1],[2],[2],[1],[1,2],[1,2],[2,1],[2,1],[1,2,1],[1,2,1]
        # dp[i][0] = total number of subsequences starting at i
        # dp[i][1] = sum of all subsequences starting at i
        dp = [[0, 0] for _ in range(len(nums))]
        mod = 10 ** 9 + 7
        dp[-1] = [1, nums[-1]]
        index_dic = {nums[-1]: [len(nums) - 1]}

        for i in range(len(nums) - 2, -1, -1):
            t = 0
            s = 0
            if nums[i] - 1 in index_dic:
                for index in index_dic[nums[i] - 1]:
                    t += dp[index][0]
                    s += dp[index][1]
            if nums[i] + 1 in index_dic:
                for index in index_dic[nums[i] + 1]:
                    t += dp[index][0]
                    s += dp[index][1]
            dp[i][0] = t + 1
            dp[i][1] = s + t * nums[i] + nums[i]
            if nums[i] not in index_dic:
                index_dic[nums[i]] = [i]
            else:
                index_dic[nums[i]].append(i)
        return sum(dp[i][1] for i in range(len(nums))) % mod


    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        # [1,2,2,1]
        # [1],[2],[2],[1],[1,2],[1,2],[2,1],[2,1],[1,2,1],[1,2,1]
        # dp[i][0] = total number of subsequences starting at i
        # dp[i][1] = sum of all subsequences starting at i
        dp = dict()
        mod = 10 ** 9 + 7
        dp[nums[-1]] = [1, nums[-1]]

        for i in range(len(nums) - 2, -1, -1):
            t = 0
            s = 0
            if nums[i] - 1 in dp:
                t += dp[nums[i] - 1][0]
                s += dp[nums[i] - 1][1]
            if nums[i] + 1 in dp:
                t += dp[nums[i] + 1][0]
                s += dp[nums[i] + 1][1]
            if nums[i] in dp:
                dp[nums[i]][0] += t + 1
                dp[nums[i]][1] += s + t * nums[i] + nums[i]
            else:
                dp[nums[i]] = [t + 1, s + t * nums[i] + nums[i]]
        return sum(dp[x][1] for x in dp) % mod

    def maxFrequency_failed(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        def find_max_left(i):
            left = 0
            right = i - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] >= nums[i] - k:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def find_max_right(i):
            left = i + 1
            right = len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= nums[i] + k:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        max_freq = 1
        for i in range(len(nums)):
            max_left = find_max_left(i)
            max_right = find_max_right(i)
            if max_left < 0 or max_left > i: max_left = i
            if max_right < i or max_right >= len(nums): max_right = i
            max_freq = max(max_freq, min(max_right - max_left + 1, numOperations + 1))
        return max_freq

    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        from collections import Counter
        counter = Counter(nums)
        def find_max_left(i):
            left = 0
            right = i - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] >= nums[i] - k:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def find_max_right(i):
            left = i + 1
            right = len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= nums[i] + k:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        def find_max_right_double(i):
            left = i
            right = len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= nums[i] + 2 * k:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        max_freq = 1
        for i in range(len(nums)):
            max_double_right = find_max_right_double(i)
            if max_double_right < i or max_double_right >= len(nums): max_double_right = i
            max_freq = max(max_freq, min(max_double_right - i + 1, numOperations))
            max_left = find_max_left(i)
            max_right = find_max_right(i)
            if max_left < 0 or max_left > i: max_left = i
            if max_right < i or max_right >= len(nums): max_right = i
            max_freq = max(max_freq, min(max_right - max_left + 1, numOperations + counter[nums[i]]))
        return max_freq

    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        freq = [0] * len(nums)
        for q in queries:
            freq[q[0]] += q[2]
            freq[q[1]] -= q[2]

        op = 0
        for i in range(len(nums)):
            op += freq[i]
            if op < nums[i]: return False
        return True

    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1: return word
        pos = []

        def largest(index):
            remaining = numFriends - index - 1
            if remaining <= 0: return word[index:]
            if index + remaining >= len(word): return ""
            return word[index:len(word) - remaining]

        for i in range(len(word)):
            pos.append(largest(i))
        pos.sort(reverse=True)
        return pos[0]

    def maximumMatchingIndices(self, nums1: List[int], nums2: List[int]) -> int:
        new_nums1 = nums1 + nums1 + nums1
        max_matching = 0
        for i in range(2 * len(nums1)):
            current = 0
            for j in range(len(nums2)):
                if new_nums1[i + j] == nums2[j]:
                    current += 1
            max_matching = max(max_matching, current)
        return max_matching

    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        from collections import Counter
        import math
        seen = set()
        counter = Counter(nums)
        keys = list(counter.keys())
        keys.sort(reverse=True)
        min_value = math.inf
        for i in keys:
            for j in range(min(min_value, i + 2 * k), i - 1, -1):
                if j not in seen:
                    seen.add(j)
                    min_value = j
                    counter[i] -= 1
                    if counter[i] == 0:
                        break
        return len(seen)

    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        counter = Counter(word2)


        total = 0
        i = 0
        new_counter = Counter()
        def check_counter():
            for c in counter:
                if new_counter[c] < counter[c]: return False
            return True

        def find_left_most_and_shrink_counter(left_most, right_most, new_counter):
            for i in range(left_most, right_most):
                if new_counter[word1[i]] - 1 >= counter[word1[i]]:
                    new_counter[word1[i]] -= 1
                else: return i
            return right_most


        while i < len(word1):
            new_counter[word1[i]] += 1
            if check_counter(): break
            i += 1

        if i == len(word1): return 0
        left_most = 0
        for j in range(i, len(word1)):
            if j != i:
                new_counter[word1[j]] += 1
            left_most = find_left_most_and_shrink_counter(left_most, j, new_counter)
            total += left_most + 1
        return total

    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues = list(set(rewardValues))
        rewardValues.sort()
        seen = set()
        seen.add(0)
        for r in rewardValues:
            new_stuff = []
            for s in seen:
                if s >= r: continue
                new_stuff.append(s + r)
            for n in new_stuff:
                seen.add(n)
        return max(seen)

    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        dp = [i for i in range(len(nums))]
        right = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] % 2 == nums[i + 1] % 2:
                right = i
            dp[i] = right

        result = []
        for q in queries:
            if dp[q[0]] >= q[1]: result.append(True)
            else: result.append(False)
        return result

    def kthLargestPerfectSubtree(self, root, k: int) -> int:
        if root is None: return -1
        subtrees = []

        def dfs(node):
            if node is None: return -1
            if node.left is None and node.right is None:
                subtrees.append(1)
                return 1
            left_size = dfs(node.left)
            right_size = dfs(node.right)
            if left_size == -1 or right_size == -1: return -1
            if left_size == right_size:
                new_size = 1 + left_size + right_size
                subtrees.append(new_size)
                return new_size
            return -1

        dfs(root)
        subtrees.sort(reverse=True)
        if k > len(subtrees): return -1
        return subtrees[k - 1]

    def countPaths(self, grid: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]

        values = []
        for i in range(m):
            for j in range(n):
                values.append((grid[i][j], i, j))
        values.sort(reverse=True)

        def neighbors(i, j):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= i + dx < m \
                    and 0 <= j + dy < n:
                    yield i + dx, j + dy

        def dfs(i, j):
            if dp[i][j] != 0: return dp[i][j]
            s = 1
            for ni, nj in neighbors(i, j):
                if grid[i][j] >= grid[ni][nj]:
                    continue
                s += dfs(ni, nj)
            dp[i][j] = s

        for _, i, j in values:
            dfs(i, j)

        return sum(sum(d) % mod for d in dp) % mod

    def waysToBuildRooms(self, prevRoom: List[int]) -> int:
        import math
        mod = 10 ** 9 + 7
        tree = [[] for _ in range(len(prevRoom))]
        for i in range(1, len(prevRoom)):
            prev = prevRoom[i]
            tree[prev].append(i)

        stack = [0]
        # 0_number of ways, 1_number of nodes
        dp = [(0, 0) for _ in range(len(prevRoom))]
        while stack:
            node = stack[-1]
            all_visited = True
            for child in tree[node]:
                if dp[child] == (0, 0):
                    all_visited = False
                    stack.append(child)

            if not all_visited: continue
            stack.pop(-1)
            if len(tree[node]) == 0:
                dp[node] = (1, 1)
                continue

            ways, child_count = dp[tree[node][0]]
            for i in range(1, len(tree[node])):
                new_ways, new_child_count = dp[tree[node][i]]
                ways = (ways * new_ways * math.comb(child_count + new_child_count, new_child_count)) % mod
                child_count += new_child_count
            dp[node] = (ways, child_count + 1)

        return dp[0][0]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3: return []
        from collections import Counter
        counter = Counter()
        counter[nums[-1]] += 1
        counter[nums[-2]] += 1

        def twoSum(target):
            result = []
            for c in counter.keys():
                counter[c] -= 1
                if target - c in counter and counter[target - c] > 0:
                    result.append((c, target - c))
                counter[c] += 1
            return result

        results = set()
        for i in range(len(nums) - 3, -1, -1):
            for r in twoSum(0 - nums[i]):
                results.add(tuple(sorted([nums[i], r[0], r[1]])))
            counter[nums[i]] += 1
        return [list(r) for r in results]

    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        trie = dict()
        END = 'END'
        for i, w in enumerate(wordsContainer):
            t = trie
            if END in t:
                if t[END][1] > len(w):
                    t[END] = (i, len(w))
            else: t[END] = (i, len(w))
            for c in reversed(w):
                if c not in t:
                    t[c] = dict()
                t = t[c]
                if END in t:
                    if t[END][1] > len(w):
                        t[END] = (i, len(w))
                else: t[END] = (i, len(w))


        results = []
        for q in wordsQuery:
            t = trie
            for c in reversed(q):
                if c not in t: break
                t = t[c]
            results.append(t[END][0])
        return results

    def remainingMethods(self, n: int, k: int, invocations: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        for inv in invocations:
            graph[inv[0]].append(inv[1])

        is_suspicious = [False] * n
        stack = [k]
        is_suspicious[k] = True
        while stack:
            node = stack.pop(-1)
            for nei in graph[node]:
                if is_suspicious[nei]: continue
                is_suspicious[nei] = True
                stack.append(nei)

        for inv in invocations:
            if is_suspicious[inv[0]]: continue
            if is_suspicious[inv[1]]: return [i for i in range(n)]

        return [i for i, s in enumerate(is_suspicious) if not s]


    def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        tree = [[] for _ in range(n)]
        for edge in edges:
            tree[edge[0]].append(edge[1])
            tree[edge[1]].append(edge[0])


        def dfs(start, end):
            visited_times = [0] * n
            seen = [False] * n
            seen[start] = True
            stack = [(start, 1 << start)]
            final_mask = 1 << start
            while stack:
                node, mask = stack.pop(-1)
                if node == end:
                    final_mask = mask
                    break
                for nei in tree[node]:
                    if seen[nei]: continue
                    seen[nei] = True
                    new_mask =  mask | (1 << nei)
                    stack.append((nei, new_mask))

            for i in range(n):
                if (final_mask & (1 << i)) != 0:
                    visited_times[i] += 1
            return visited_times

        total_visited_times = []
        for t in trips:
            total_visited_times.append(dfs(t[0], t[1]))

        costs = [price[i] * sum(total_visited_times[j][i] for j in range(len(total_visited_times))) for i in range(n)]

        import math
        # 0 - half, 1 - original
        dp = [[math.inf, math.inf] for _ in range(n)]

        seen = [False] * n
        def dfs2(node, half_or_not):
            if seen[node]:
                return dp[node][half_or_not]
            seen[node] = True

            half_cost = costs[node] // 2
            original_cost = costs[node]

            for nei in tree[node]:
                if seen[nei]: continue
                v1 = dfs2(nei, 1)
                v2 = dfs2(nei, 0)
                half_cost += v1
                original_cost += min(v1, v2)
            dp[node][0] = half_cost
            dp[node][1] = original_cost
            return dp[node][half_or_not]

        return min(dfs2(0, 0), dfs2(0, 1))


    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import math
        order = [(damage[i] / math.ceil(health[i] / power), i) for i in range(len(damage))]
        order.sort(reverse=True)
        total_damage = sum(damage)
        total = 0
        for _, i in order:
            total += total_damage * math.ceil(health[i] / power)
            total_damage -= damage[i]
        return total

    def calculateScore(self, s: str) -> int:
        def get_mirror(c):
            return chr((ord('z') - ord('a')) - (ord(c) - ord('a')) + ord('a'))

        character_to_index = dict()
        score = 0
        for i, c in enumerate(s):
            mirror_c = get_mirror(c)
            if mirror_c not in character_to_index or len(character_to_index[mirror_c]) == 0:
                if c not in character_to_index:
                    character_to_index[c] = [i]
                else:
                    character_to_index[c].append(i)
            else:
                closest_index = character_to_index[mirror_c][-1]
                character_to_index[mirror_c].pop(-1)
                score += i - closest_index
        return score

    def getSum(self, nums: List[int]) -> int:
        # key: number, value: [sum, times]
        mod = 10 ** 9 + 7
        def get_count(is_decreasing):
            numbers = dict()
            for n in nums:
                if is_decreasing:
                    next_value = n - 1
                else:
                    next_value = n + 1

                if next_value in numbers:
                    previous_value = numbers[next_value]
                    if n not in numbers:
                        numbers[n] = [(n + n * previous_value[1] + previous_value[0]) % mod, 1 + previous_value[1]]
                    else:
                        current_value = numbers[n]
                        numbers[n] = [(current_value[0] + n + n * previous_value[1] + previous_value[0]) % mod,
                                      1 + current_value[1] + previous_value[1]]
                else:
                    if n not in numbers:
                        numbers[n] = [n, 1]
                    else:
                        original_value = numbers[n]
                        numbers[n] = [(original_value[0] + n) % mod, original_value[1] + 1]
            return sum(v[0] for k, v in numbers.items()) % mod
        return (get_count(True) + get_count(False) - sum(nums)) % mod

    def countPairs_failed(self, n: int, edges: List[List[int]], queries: List[int]) -> List[int]:
        import bisect
        from collections import Counter
        totals = [0 for _ in range(n)]
        edge_counts = [Counter() for _ in range(n)]
        for e in edges:
            edge_counts[e[0] - 1][e[1] - 1] += 1
            edge_counts[e[1] - 1][e[0] - 1] += 1
            totals[e[0] - 1] += 1
            totals[e[1] - 1] += 1

        results = [0 for _ in range(len(queries))]
        for i in range(1, n):
            for j in range(i):
                result = totals[i] + totals[j] - edge_counts[i][j]
                for k, q in enumerate(queries):
                    if result > q:
                        results[k] += 1
        return results

    def maxIncreasingSubarrays(self, nums: List[int]) -> int:
        max_decreasing = [1 for _ in range(len(nums))]
        max_increasing = [1 for _ in range(len(nums))]

        consecutively_increasing = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                consecutively_increasing += 1
            else:
                consecutively_increasing = 1
            max_decreasing[i] = consecutively_increasing

        consecutively_decreasing = 1
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                consecutively_decreasing += 1
            else:
                consecutively_decreasing = 1
            max_increasing[i] = consecutively_decreasing

        result = 1
        for i in range(1, len(nums)):
            result = max(min(max_decreasing[i - 1], max_increasing[i]), result)
        return result

    def maxProfit(self, prices: List[int], profits: List[int]) -> int:
        N = len(prices)
        max_left = [-1 for _ in range(N)]
        for i in range(N):
            for j in range(i):
                if prices[j] < prices[i]:
                    max_left[i] = max(max_left[i], profits[j])
        max_right = [-1 for _ in range(N)]
        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N):
                if prices[j] > prices[i]:
                    max_right[i] = max(max_right[i], profits[j])

        result = -1
        for i in range(N):
            if max_left[i] == -1 or max_right[i] == -1: continue
            result = max(result, max_right[i] + max_left[i] + profits[i])
        return result



    def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
        import math


        class SegmentTree:
            def __init__(self, N):
                self.arr = [-1] * (2 * N)
                self.N = N
                for i in range(self.N):
                    self.arr[i + self.N] = math.inf
                for i in range(self.N - 1, 0, -1):
                    self.arr[i] = self.merge(self.arr[i << 1], self.arr[(i << 1) + 1])

            def merge(self, x, y):
                return min(x, y)

            def query(self, l, r):
                # [l, r)
                l += self.N
                r += self.N

                result = math.inf
                while l < r:
                    if l & 1:
                        result = min(result, self.arr[l])
                        l += 1
                    if r & 1:
                        r -= 1
                        result = min(result, self.arr[r])
                    l = l >> 1
                    r = r >> 1
                if result == math.inf: return -1
                return result

            def update(self, i, value):
                i = i + self.N
                self.arr[i] = min(self.arr[i], value)

                while i > 0:
                    if i & 1:
                        value = self.merge(self.arr[i - 1], self.arr[i])
                    else:
                        value = self.merge(self.arr[i + 1], self.arr[i])
                    i = i >> 1
                    self.arr[i] = value

        mapping = dict()
        new_heights = set()
        for h in heights:
            new_heights.add(h)
            new_heights.add(h + 1)

        for h in sorted(new_heights):
            if h in mapping: continue
            mapping[h] = len(mapping)

        queries = [(min(q[0], q[1]), max(q[0], q[1]), i) for i, q in enumerate(queries)]
        queries.sort(key=lambda x: max(x[0], x[1]), reverse=True)
        tree = SegmentTree(len(mapping))
        index = len(heights) - 1

        results = [-1] * len(queries)
        for q in queries:
            if q[0] == q[1]:
                results[q[2]] = q[0]
                continue
            if heights[q[1]] > heights[q[0]]:
                results[q[2]] = q[1]
                continue
            left = max(heights[q[0]], heights[q[1]]) + 1
            while index >= max(q[0], q[1]):
                tree.update(mapping[heights[index]], index)
                index -= 1
            results[q[2]] = tree.query(mapping[left], len(mapping))
        return results

    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        col_dict = dict()
        stack = [(root, 0, 0)]
        while stack:
            node, row, col = stack.pop(-1)
            if col not in col_dict:
                col_dict[col] = {row: [node.val]}
            else:
                if row not in col_dict[col]:
                    col_dict[col][row] = [node.val]
                else: col_dict[col][row].append(node.val)

            if node.left is not None:
                stack.append((node.left, row + 1, col - 1))
            if node.right is not None:
                stack.append((node.right, row + 1, col + 1))

        final_results = []
        for col in sorted(col_dict.keys()):
            row_dict = col_dict[col]
            results = []
            for k in sorted(row_dict.keys()):
                results.extend(sorted(row_dict[k]))
            final_results.append(results)
        return final_results


    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10 ** 9 + 7
        dp = [0] * (k + 1)
        for i in range(n):
            new_dp = [0] * (k + 1)
            new_dp[0] = (dp[0] * (m - 1)) % mod
            for j in range(1, k + 1):
                new_dp[j] = (dp[j] * (m - 1) + dp[j - 1]) % mod
            dp = new_dp
        return dp[-1]

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        import heapq
        heap = []

        all_current = [node for node in lists]
        for i, node in enumerate(lists):
            if node is not None:
                heapq.heappush(heap, (node.val, i))

        if not heap: return None
        head = ListNode(val=heap[0][0])
        all_current[heap[0][1]] = all_current[heap[0][1]].next
        if all_current[heap[0][1]] is not None:
            heapq.heappush(heap, (all_current[heap[0][1]].val, heap[0][1]))
        heapq.heappop(heap)

        prev = head
        while heap:
            val, index = heapq.heappop(heap)
            current = ListNode(val)
            prev.next = current
            prev = current
            all_current[index] = all_current[index].next
            if all_current[index] is not None:
                heapq.heappush(heap, (all_current[index].val, index))
        return head

    def minLength(self, s: str, numOps: int) -> int:
        other_c = {'1':'0', '0':'1'}
        def check(n):
            prev_c = None
            prev_count = 0
            ops_count = numOps
            for i, c in enumerate(s):
                if prev_c is None:
                    prev_count = 1
                    prev_c = c
                elif prev_c != c:
                    prev_count = 1
                    prev_c = c
                else:
                    prev_count += 1

                if prev_count > n:
                    if ops_count <= 0: return False
                    ops_count -= 1
                    if i + 1 < len(s):
                        # 1000-x
                        if prev_count > 2:
                            # 1000-0
                            if s[i + 1] == c:
                                prev_c = other_c[c]
                            # 1000-1
                            else:
                                prev_c = c
                        # 100-x
                        else:
                            # 100-0
                            if s[i + 1] == c:
                                prev_c = other_c[c]
                            # 100-1
                            else:
                                if i - 1 == 0:
                                    prev_c = c
                                else:
                                    return False
                    prev_count = 1
            return True

        left, right = 1, len(s)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left

    def isRobotBounded(self, instructions: str) -> bool:
        from collections import Counter

        x, y = 0, 0
        direction = (1, 0)
        left_next_direction = {(1, 0):(0, 1), (0, 1):(-1, 0), (-1, 0):(0, -1), (0, -1):(1, 0)}
        right_next_direction = {(1, 0):(0, -1), (0, -1):(-1, 0), (-1, 0):(0, 1), (0, 1):(1, 0)}
        for inst in instructions:
            if inst == 'G':
                x += direction[0]
                y += direction[1]
            elif inst == 'L':
                direction = left_next_direction[direction]
            else:
                direction = right_next_direction[direction]
        if x == 0 and y == 0: return True

        counter = Counter(instructions)
        counter['L'] = counter['L'] % 4
        counter['R'] = counter['R'] % 4
        if counter['L'] != counter['R']:
            return True
        else: return False

    def largestRectangleArea(self, heights: List[int]) -> int:
        class SegmentTree:
            def __init__(self, n, merge_function, default_value):
                self.default = default_value
                self.N = n
                self.merge = merge_function
                self.arr = [default_value] * 2 * self.N

            def update(self, index, value):
                index = self.N + index
                self.arr[index] = value
                index = index >> 1
                while index > 0:
                    self.arr[index] = self.merge(self.arr[(index << 1)], self.arr[(index << 1) + 1])
                    index = index >> 1

            def query(self, left, right):
                l, r = left + self.N, right + self.N
                result = self.default
                while l < r:
                    if l % 2 == 1:
                        result = self.merge(result, self.arr[l])
                        l += 1
                    if r % 2 == 1:
                        r -= 1
                        result = self.merge(result, self.arr[r])
                    l = l >> 1
                    r = r >> 1
                return result

        max_tree = SegmentTree(len(heights), lambda x, y: max(x, y), -1)
        min_tree = SegmentTree(len(heights), lambda x, y: min(x, y), len(heights))
        result = 0

        sorted_heights = [(h, i) for i, h in enumerate(heights)]
        sorted_heights.sort()

        for h, i in sorted_heights:
            max_left = max_tree.query(0, i + 1)
            min_right = min_tree.query(i, len(heights))
            max_tree.update(i, i)
            min_tree.update(i, i)
            result = max(result, (min_right - max_left - 1) * h)
        return result

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        result = []
        import heapq
        heap = []
        index = 0
        for i in range(k):
            heapq.heappush(heap, (-nums[index], index))
            index += 1

        result.append(-heap[0][0])
        while index < len(nums):
            heapq.heappush(heap, (-nums[index], index))
            while heap[0][1] <= index - k:
                heapq.heappop(heap)
            result.append(-heap[0][0])
            index += 1
        return result

    def subarraySum(self, nums: List[int], k: int) -> int:
        from collections import Counter
        running_sum = 0
        counter = Counter()
        counter[0] = 1
        result = 0
        for n in nums:
            running_sum += n
            result += counter[running_sum - k]
            counter[running_sum] += 1
        return result

    def maxSubArray(self, nums: List[int]) -> int:
        result = min(nums)
        min_sum = 0
        running_sum = 0
        for n in nums:
            running_sum += n
            result = max(result, running_sum - min_sum)
            min_sum = min(min_sum, running_sum)
        return result

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        from collections import Counter
        def gen_rows(i):
            for j in range(9):
                yield i, j

        def gen_cols(j):
            for i in range(9):
                yield i, j

        def gen_square(i, j):
            for k1 in range(3):
                for k2 in range(3):
                    yield i + k1, j + k2

        def check(indices):
            counter = Counter()
            for i, j in indices:
                counter[board[i][j]] += 1
            for i in range(1, 10):
                if counter[str(i)] > 1: return False
            return True

        for i in range(9):
            if not check(list(gen_cols(i))):
                return False
            if not check(list(gen_rows(i))):
                return False

        for i in [0, 3, 6]:
            for j in [0, 3, 6]:
                if not check(list(gen_square(i, j))):
                    return False
        return True

    def longestPalindrome(self, s: str) -> str:
        dp = [[0 for _ in range(len(s))] for _ in range(len(s))]

        for i in range(len(s)):
            dp[i][i] = 1

        for k in range(2, len(s) + 1):
            for i in range(len(s) - k + 1):
                if s[i] != s[i + k - 1]:
                    dp[i][i + k - 1] = 0
                else:
                    if k == 2: dp[i][i + k - 1] = 2
                    elif dp[i + 1][i + k - 2] != 0:
                        dp[i][i + k - 1] = 2 + dp[i + 1][i + k - 2]
        left, right = 0, 0
        for k in range(1, len(s) + 1):
            for i in range(len(s) - k + 1):
                if dp[i][i + k - 1] > right - left + 1:
                    left = i
                    right = i + k - 1
        return s[left:right + 1]

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        import math
        if len(nums1) < len(nums2):
            nums1, nums2 = nums2, nums1
        lower_size = (len(nums1) + len(nums2)) // 2

        def check(index):
            index2 = lower_size - index - 2
            if 0 <= index2 + 1 < len(nums2):
                return nums1[index] <= nums2[index2 + 1]
            elif index2 + 1 >= len(nums2):
                return True
            else:
                False


        left, right = 0, len(nums1) - 1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1

        index2 = lower_size - left - 1
        if (len(nums1) + len(nums2)) % 2 == 1:
            if index2 + 1 < len(nums2):
                return min(nums2[index2 + 1], nums1[left])
            else: return nums1[left]
        else:
            if index2 < 0:
                lower_nums2 = -math.inf
            else: lower_nums2 = nums2[index2]
            if index2 + 1 >= len(nums2):
                upper_nums2 = math.inf
            else:
                upper_nums2 = nums2[index2 + 1]
            if left >= len(nums1):
                upper_nums1 = math.inf
            else:
                upper_nums1 = nums1[left]
            if left - 1 < 0:
                lower_nums1 = -math.inf
            else: lower_nums1 = nums1[left - 1]
            return (max(lower_nums1, lower_nums2) + min(upper_nums1, upper_nums2)) / 2

    def simplifyPath(self, path: str) -> str:
        split_paths = path.split("/")
        cleaned = []
        for s in split_paths:
            if s == "/": continue
            cleaned.append(s)
        stack = []
        for c in cleaned:
            if c == "..":
                if stack: stack.pop(-1)
            elif c == ".":
                continue
            elif c:
                stack.append(c)
        return "/" + "/".join(stack)

    def firstMissingPositive(self, nums: List[int]) -> int:
        def swap(i):
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]

        for i in range(len(nums)):
            while True:
                if nums[i] < 0 or nums[i] >= len(nums): break
                if nums[i] == i: break
                if nums[i] == nums[nums[i]]: break
                swap(i)

        for i in range(1, len(nums)):
            if nums[i] != i and nums[0] != i: return i
        if nums[0] != len(nums): return len(nums)
        else: return len(nums) + 1

    def maxPoints(self, points: List[List[int]]) -> int:
        import math
        from collections import Counter
        if len(points) <= 1: return len(points)
        def get_function(x1, y1, x2, y2):
            # return a function (a,b) where y = ax + b
            # y1 = ax1 + b
            # y2 = ax2 + b
            if (x1-x2) == 0:
                a = math.inf
                b = -x1
            else:
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1
            return (a, b)

        counter = Counter()
        sets = [set() for _ in range(len(points))]
        for i in range(1, len(points)):
            for j in range(i):
                a, b = get_function(points[i][0], points[i][1], points[j][0], points[j][1])
                sets[i].add((a, b))
                sets[j].add((a, b))
        for s in sets:
            for a, b in s:
                counter[(a, b)] += 1
        return max(counter.values())

    def findKthLargest(self, nums: List[int], k: int) -> int:
        import heapq
        heap = []
        index = 0
        for i in range(k):
            heap.append(nums[i])
            index += 1
        heapq.heapify(heap)
        while index < len(nums):
            if nums[index] > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, nums[index])
            index += 1
        return heap[0]

    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        def get_new_c(moves, i):
            return chr(ord('a') + (ord(s[i]) - ord('a') + moves) % 26)

        def translate(shift):
            if shift == 0: return -1
            else: return 1

        result = []
        moves_arr = [0] * (len(s) + 1)
        for shift in shifts:
            moves_arr[shift[0]] += translate(shift[2])
            moves_arr[shift[1] + 1] -= translate(shift[2])

        accumulative_moves = 0
        for i in range(len(s)):
            accumulative_moves += moves_arr[i]
            result.append(get_new_c(accumulative_moves, i))
        return ''.join(result)

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        if len(nums) <= 1: return False
        d = dict()
        running_sum = nums[0]
        d[running_sum % k] = 0
        d[0] = -1

        for i in range(1, len(nums)):
            n = nums[i]
            running_sum += n
            x = running_sum % k
            if x in d:
                last_index = d[x]
                if i - last_index > 1:
                    return True
            else:
                d[x] = i
        return False

    def isValidPalindrome(self, s: str, k: int) -> bool:
        import math
        dp = [[math.inf for _ in range(len(s))] for _ in range(len(s))]

        for length in range(1, len(s) + 1):
            for i in range(len(s) - length + 1):
                j = i + length - 1
                if i == j:
                    dp[i][j] = 0
                    continue
                if s[i] == s[j]:
                    if length == 2:
                        dp[i][j] = 0
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1)
        return dp[0][-1] <= k

    def largestIsland(self, grid: List[List[int]]) -> int:
        M = len(grid)
        N = len(grid[0])
        if M == 0 or N == 0: return 0
        def gen(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if x + dx < 0 or x + dx >= M: continue
                if y + dy < 0 or y + dy >= N: continue
                yield x + dx, y + dy

        groups = [[(i, j, 1) for j in range(N)] for i in range(M)]
        def find(i, j):
            g = groups[i][j]
            if g[0] == i and g[1] == j: return g
            else:
                g = find(g[0], g[1])
                groups[i][j] = g
                return g

        def union(i1, j1, i2, j2):
            g1 = find(i1, j1)
            g2 = find(i2, j2)
            if g1[0] == g2[0] and g1[1] == g2[1]: return
            if g1[2] > g2[2]:
                groups[g1[0]][g1[1]] = (g1[0], g1[1], g1[2] + g2[2])
                groups[g2[0]][g2[1]] = (g1[0], g1[1], g1[2] + g2[2])
            else:
                groups[g2[0]][g2[1]] = (g2[0], g2[1], g1[2] + g2[2])
                groups[g1[0]][g1[1]] = (g2[0], g2[1], g1[2] + g2[2])

        for i in range(M):
            for j in range(N):
                if grid[i][j] == 0: continue
                for ni, nj in gen(i, j):
                    if grid[ni][nj] == 0: continue
                    union(i, j, ni, nj)

        result = 0
        for i in range(M):
            for j in range(N):
                result = max(result, groups[i][j][2])
                if grid[i][j] == 1: continue
                adj_groups = set()
                for ni, nj in gen(i, j):
                    if grid[ni][nj] == 0: continue
                    g = find(ni, nj)
                    adj_groups.add(g)
                result = max(result, sum([g[2] for g in adj_groups]) + 1)
        return result

    def shortestDistance(self, grid: List[List[int]]) -> int:
        from collections import deque
        import math
        M = len(grid)
        N = len(grid[0])
        distances = [[[0, 0] for _ in range(N)] for _ in range(M)]

        def gen(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if x + dx < 0 or x + dx >= M: continue
                if y + dy < 0 or y + dy >= N: continue
                if grid[x + dx][y + dy] == 1 or grid[x + dx][y + dy] == 2: continue
                yield x + dx, y + dy

        def BFS(i, j):
            seen = [[False for _ in range(N)] for _ in range(M)]
            seen[i][j] = True
            queue = deque()
            queue.append((i, j, 0))
            while queue:
                x, y, d = queue.popleft()
                if grid[x][y] == 0:
                    distances[x][y][0] += d
                    distances[x][y][1] += 1
                for nx, ny in gen(x, y):
                    if seen[nx][ny]: continue
                    seen[nx][ny] = True
                    queue.append((nx, ny, d + 1))

        building_count = 0
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    building_count += 1
                    BFS(i, j)
        result = math.inf
        for i in range(M):
            for j in range(N):
                if grid[i][j] != 0: continue
                if distances[i][j][1] != building_count: continue
                result = min(result, distances[i][j][0])
        if result == math.inf: return -1
        return result

    def removeInvalidParentheses(self, s: str) -> List[str]:
        def expand(arr, size):
            while len(arr) < size:
                arr.append(set())

        lefts = [{""}]
        for c in s:
            new_lefts = [{""}, set()]
            for i, s in enumerate(lefts):
                expand(new_lefts, i + 2)
                for sequence in s:
                    if c == '(':
                        new_lefts[i + 1].add(sequence + '(')
                        new_lefts[i].add(sequence)
                    elif c == ')':
                        if i > 0:
                            new_lefts[i - 1].add(sequence + ')')
                        new_lefts[i].add(sequence)
                    else:
                        new_lefts[i].add(sequence + c)
            lefts = new_lefts

        d = dict()
        for c in lefts[0]:
            if len(c) not in d:
                d[len(c)] = [c]
            else:
                d[len(c)].append(c)
        if not d: return []
        return d[max(d.keys())]


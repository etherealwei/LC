from collections import defaultdict

from typing import *

from leetcode_0_types import *


class Solution:
    def flatten(self, head):
        def recur(head):
            node = head
            prev = None

            while node is not None:
                if node.child is not None:
                    child_head, child_tail = recur(node.child)
                    node.child = None
                    child_tail.next = node.next
                    node.next = child_head
                    child_head.prev = node
                    prev = child_tail
                    node = prev.next
                    if node is not None:
                        node.prev = prev
                    else:
                        prev = node
                        node = node.next
            return head, prev

        recur(head)
        return head

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        node = root

        while True:
            if p.val == node.val or q.val == node.val: return node
            if p.val < node.val and q.val < node.val:
                node = node.left
            elif p.val > node.val and q.val > node.val:
                node = node.right
            else:
                return node

    def minimumDifferenceTLE(self, nums: List[int]) -> int:
        import math
        seen = dict()
        nums_total = sum(nums)
        N = len(nums)

        def recur(mask, count, total):
            # mask - all numbers selected in the left partition
            # return min abs difference if we are to select N//2 - count more numbers into the left partition
            if mask in seen: return seen[mask]
            if count == N // 2:
                r = abs(nums_total - 2 * total)
                seen[mask] = r
                return r

            min_diff = math.inf
            for i in range(N):
                if (mask & (1 << i)) != 0: continue
                new_mask = mask | (1 << i)
                min_diff = min(min_diff, recur(new_mask, count + 1, total + nums[i]))
            seen[mask] = min_diff
            return min_diff

        return recur(0, 0, 0)

    def minimumDifference(self, nums: List[int]) -> int:
        import math
        N = len(nums) // 2
        left_nums = nums[:N]
        left_total = sum(left_nums)
        right_nums = nums[N:]
        right_total = sum(right_nums)

        right_seen = dict()

        def recur_right(i, count, total):
            if count not in right_seen:
                right_seen[count] = set()
            right_seen[count].add(total)
            if i < N:
                recur_right(i + 1, count + 1, total + right_nums[i])
                recur_right(i + 1, count, total)

        recur_right(0, 0, 0)
        for k in right_seen:
            right_seen[k] = list(sorted(right_seen[k]))

        def get_right(left_count, total):
            arr = right_seen[N - left_count]
            left, right = 0, len(arr) - 1
            target = (left_total + right_total) / 2 - total
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            possibles = []
            if left < len(arr):
                possibles.append(arr[left])
            if left - 1 >= 0:
                possibles.append(arr[left - 1])
            min_dff = math.inf
            for p in possibles:
                min_dff = min(min_dff, abs(left_total - total + right_total - p - (p + total)))
            return min_dff


        def recur_left(i, count, total):
            # mask - all numbers selected in the left partition
            # return min abs difference if we are to select N//2 - count more numbers into the left partition
            min_diff = math.inf
            if i < N:
                min_diff = min(min_diff, recur_left(i + 1, count + 1, total + left_nums[i]))
                min_diff = min(min_diff, recur_left(i + 1, count, total))
            min_diff = min(min_diff, get_right(count, total))
            return min_diff

        return recur_left(0, 0, 0)

    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        from collections import Counter
        invalid_ones = []
        tdict = dict()
        for i, tra in enumerate(transactions):
            t = tra.split(',')
            if t[0] not in tdict:
                tdict[t[0]] = [(int(t[1]), t[-1], int(t[2]), i)]
            else:
                tdict[t[0]].append((int(t[1]), t[-1], int(t[2]), i))

        for _, arr in tdict.items():
            arr.sort()
            added = [False] * len(arr)
            for i in range(len(arr)):
                if arr[i][2] > 1000:
                    added[i] = True

            counter = Counter()
            left = 0
            counter[arr[left][1]] += 1
            for i in range(1, len(arr)):
                counter[arr[i][1]] += 1
                while left < i and arr[i][0] - arr[left][0] > 60:
                    counter[arr[left][1]] -= 1
                    if counter[arr[left][1]] == 0:
                        del counter[arr[left][1]]
                    left += 1
                if len(counter) > 1:
                    added[i] = True

            counter = Counter()
            right = len(arr) - 1
            counter[arr[right][1]] += 1
            for i in range(len(arr) - 2, -1, -1):
                counter[arr[i][1]] += 1
                while right > i and arr[right][0] - arr[i][0] > 60:
                    counter[arr[right][1]] -= 1
                    if counter[arr[right][1]] == 0:
                        del counter[arr[right][1]]
                    right -= 1
                if len(counter) > 1:
                    added[i] = True

            for i, a in enumerate(added):
                if a:
                    invalid_ones.append(transactions[arr[i][-1]])

        return invalid_ones

    def punishmentNumber(self, n: int) -> int:
        def can_get_target(s, target):
            if not s: return target == 0
            for i in range(len(s)):
                x = int(s[:i + 1])
                if can_get_target(s[i + 1:], target - x): return True
            return False


        def is_punishment(n):
            s = str(n * n)
            return can_get_target(s, n)

        total = 0
        for i in range(1, n + 1):
            if is_punishment(i): total += i * i
        return total

    def getHappyString(self, n: int, k: int) -> str:
        import math
        letters = ['a', 'b', 'c']
        total = 3 * (2 ** (n - 1))
        if k > total: return ''
        top = math.ceil(k / (2 ** (n - 1))) - 1


        result = [letters[top]]

        def pick_letter(prev_letter, top):
            for i in range(3):
                if i == prev_letter: continue
                top -= 1
                if top < 0: return i


        prev_letter = top
        k =  k % (2 ** (n - 1))
        n -= 1
        while n > 0:
            if k == 0:
                top = 1
            else:
                top = math.ceil(k / (2 ** (n - 1))) - 1
            top = pick_letter(prev_letter, top)
            result.append(letters[top])
            prev_letter = top
            k =  k % (2 ** (n - 1))
            n -= 1
        return ''.join(result)

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []

        def recur(i, path, total):
            if total == target:
                result.append([p for p in path])
                return
            if total > target or i >= len(candidates): return
            recur(i + 1, path, total)
            for k in range(1, (target - total) // candidates[i] + 1):
                path.append(candidates[i])
                recur(i + 1, path, total + k * candidates[i])
            for k in range(1, (target - total) // candidates[i] + 1):
                path.pop(-1)
        recur(0, [], 0)
        return result

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result = []
        def recur(i, total, path):
            if total == n:
                if len(path) == k:
                    result.append([p for p in path])
                    return
                else: return
            if i >= 10: return
            path.append(i)
            recur(i + 1, total + i, path)
            path.pop(-1)
            recur(i + 1, total, path)
        recur(1, 0, [])
        return result

    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        import math
        levels = []

        def expand_levels(current):
            while len(levels) <= current: levels.append([math.inf, -math.inf])

        def recur(node, level, index):
            expand_levels(level)
            levels[level][0] = min(levels[level][0], index)
            levels[level][1] = max(levels[level][1], index)
            if node.left is not None:
                recur(node.left, level + 1, 2 * index)
            if node.right is not None:
                recur(node.right, level + 1, 2 * index + 1)
        recur(root, 0, 0)
        return max(x[1] - x[0] + 1 for x in levels)

    def checkPowersOfThree(self, n: int) -> bool:
        def check(x, total):
            if total > n: return False
            if total == n: return True
            if x > n: return False
            if check(x * 3, total): return True
            if check(x * 3, total + x): return True
            return False
        return check(1, 0)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def recur(preorder_i, inorder_i, parent_val):
            node = TreeNode(preorder[preorder_i])

            if preorder_i + 1 >= len(preorder): return node, preorder_i, inorder_i
            if preorder[preorder_i] != inorder[inorder_i]:
                node.left, preorder_i, inorder_i = recur(preorder_i + 1, inorder_i, node.val)

            if preorder_i + 1 >= len(preorder): return node, preorder_i, inorder_i
            if parent_val is None or inorder[inorder_i + 1] != parent_val:
                node.right, preorder_i, inorder_i = recur(preorder_i + 1, inorder_i + 1, parent_val)
            else: inorder_i += 1
            return node, preorder_i, inorder_i
        return recur(0, 0, None)[0]

    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        if left < len(nums) and nums[left] == target: return left
        return -1

    def pivotArrayNoRelativeOrder(self, nums: List[int], pivot: int) -> List[int]:
        left = 0
        right = len(nums) - 1
        while left < right:
            while left < right and nums[left] == pivot:
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
            left += 1

        length = left

        def partition(nums, length):
            # return index > pivot
            left, right = 0, length - 1
            while left < right:
                if nums[left] > pivot and nums[right] < pivot:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right += 1
                elif nums[left] > pivot and nums[right] > pivot:
                    right -= 1
                elif nums[left] < pivot and nums[right] > pivot:
                    left += 1
                    right += 1
                else:
                    left += 1
            if left >= length or nums[left] > pivot: return left
            return left + 1

        partition(nums, length)
        left = 0
        while left < len(nums):
            if nums[left] > pivot: break
            left += 1
        if left == len(nums): return nums

        left, right = left, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        return nums

    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        result = [0] * len(nums)
        i = 0
        pivot_count = 0
        for n in nums:
            if n < pivot:
                result[i] = n
                i += 1
            elif n == pivot:
                pivot_count += 1

        for _ in range(pivot_count):
            result[i] = pivot
            i += 1

        for n in nums:
            if n > pivot:
                result[i] = n
                i += 1
        return result

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        from collections import Counter
        s_counter = Counter(s)
        t_counter = Counter(t)

        for k, v in s_counter.items():
            if t_counter[k] != v: return False
        return True

    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        count = 1
        for i in range(n):
            matrix[0][i] = count
            count += 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        index = 1
        length = n - 1
        i, j = 0, n - 1
        while length > 0:
            di, dj = directions[index]
            for _ in range(length):
                i += di
                j += dj
                matrix[i][j] = count
                count += 1
            index = (index + 1) % 4
            if index == 3 or index == 1:
                length -= 1
        return matrix

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None: return head
        node = head
        count = 0
        tail = None
        while node is not None:
            tail = node
            node = node.next
            count += 1

        k = (count - k % count) % count
        node = head
        prev = None
        while k > 0:
            prev = node
            node = node.next
            k -= 1

        if prev is not None:
            prev.next = None
            tail.next = head
            return node
        else: return head

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1: return 0
        M = len(obstacleGrid)
        N = len(obstacleGrid[0])
        dp = [[0 for _ in range(N)] for _ in range(M)]
        dp[-1][-1] = 1

        for i in range(M - 1, -1, -1):
            for j in range(N - 1, -1, -1):
                if i == M - 1 and j == N - 1: continue
                if obstacleGrid[i][j] == 1: continue
                if i + 1 < M and obstacleGrid[i + 1][j] != 1:
                    dp[i][j] += dp[i + 1][j]
                if j + 1 < N and obstacleGrid[i][j + 1] != 1:
                    dp[i][j] += dp[i][j + 1]
        return dp[0][0]

    def plusOne(self, digits: List[int]) -> List[int]:
        i = len(digits) - 1
        remain = 1
        while i >= 0:
            if not remain: break
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                remain = 0
            i -= 1
        if remain != 0:
            return [1] + digits
        else: return digits

    def addBinary(self, a: str, b: str) -> str:
        if len(a) < len(b):
            return self.addBinary(b, a)
        b = '0' * (len(a) - len(b)) + b
        i = len(a) - 1
        remain = 0
        result = ''
        while i >= 0:
            x = remain + int(a[i]) + int(b[i])
            result = str(x % 2) + result
            remain = x // 2
            i -= 1
        if remain != 0:
            result = '1' + result
        return result

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        M = len(matrix)
        N = len(matrix[0])

        def find_row():
            left, right = 0, M - 1
            while left <= right:
                mid = (left + right) // 2
                if target > matrix[mid][-1]:
                    left = mid + 1
                elif target < matrix[mid][0]:
                    right = mid - 1
                else:
                    return mid
            return None
        row = find_row()
        if row is None: return False
        left, right = 0, N - 1
        while left <= right:
            mid = (left + right) // 2
            if target > matrix[row][mid]:
                left = mid + 1
            else:
                right = mid - 1
        if left == N: return False
        return matrix[row][left] == target

    def minDistance(self, word1: str, word2: str) -> int:
        #  S, SE
        # [0, 1
        #  1, 0
        #  0, 0]
        import math
        if not word1: return len(word2)
        if not word2: return len(word1)

        dp = [[math.inf for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]
        dp[-1][-1] = 0
        for i in range(len(word1)):
            dp[i][-1] = len(word1) - i
        for j in range(len(word2)):
            dp[-1][j] = len(word2) - j

        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2) - 1, -1, -1):
                if word1[i] == word2[j]:
                    dp[i][j] = min(dp[i + 1][j + 1], 1 + dp[i][j + 1], 1 + dp[i + 1][j])
                else:
                    dp[i][j] = min(1 + dp[i][j + 1], 1 + dp[i + 1][j], 1 + dp[i + 1][j + 1])
        return dp[0][0]

    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []

        def backtrack(start, arr):
            if len(arr) == k:
                result.append(arr[:])
                return
            for i in range(start, n + 1):
                arr.append(i)
                backtrack(start + 1, arr)
                arr.pop()
        backtrack(1, [])
        return result

    def subsets(self, nums: List[int]) -> List[List[int]]:

        def recur(start):
            if start == len(nums) - 1:
                return [[], [nums[-1]]]
            results = []
            subsets = recur(start + 1)
            for s in subsets:
                results.append(s)
                results.append(s + [nums[start]])
            return results
        return recur(0)

    def exist(self, board: List[List[str]], word: str) -> bool:
        M = len(board)
        N = len(board[0])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        def gen(i, j):
            for di, dj in dirs:
                if i + di < 0 or i + di >= M: continue
                if j + dj < 0 or j + dj >= N: continue
                yield i + di, j + dj

        def search(i, j, start):
            if board[i][j] != word[start]: return False
            if start == len(word) - 1: return True

            original = board[i][j]
            board[i][j] = '#'
            for ni, nj in gen(i, j):
                if search(ni, nj, start + 1): return True
            board[i][j] = original
            return False

        for i in range(M):
            for j in range(N):
                if search(i, j, 0): return True
        return False

    def removeDuplicates(self, nums: List[int]) -> int:
        read = 1
        write = 1
        count = 1
        while read < len(nums):
            if nums[read] == nums[read - 1]:
                count += 1
            else: count = 1
            if count <= 2:
                nums[write] = nums[read]
                write += 1
            read += 1
        return write

    def search(self, nums: List[int], target: int) -> bool:
        if not nums: return False

        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if target == nums[left] or target == nums[right] or target == nums[mid]: return True
            if nums[left] == nums[mid]:
                left += 1
                continue
            if nums[left] < nums[mid]:
                # left and mid are in the same side
                if target > nums[mid]:
                    left = mid + 1
                elif target > nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # left in higher and mid in lower
                if target > nums[left]:
                    right = mid - 1
                elif target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        import math
        dummy = ListNode(math.inf)
        prev_valid = dummy
        prev_valid.next = head
        prev = None
        node = head
        count = 0
        while node is not None:
            if prev is None:
                count = 1
            elif prev.val != node.val:
                if count > 1:
                    prev_valid.next = node
                else:
                    prev_valid = prev
                count = 1
            else:
                count += 1
            prev = node
            node = node.next

        if count > 1:
            prev_valid.next = None
        return dummy.next

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        from collections import Counter
        counter = Counter(nums)
        keys = list(counter.keys())

        def recur(start):
            if start >= len(keys): return [[]]
            subsets = recur(start + 1)
            result = []
            for s in subsets:
                result.append(s)
                for i in range(1, counter[keys[start]] + 1):
                    result.append([keys[start]] * i + s)
            return result
        return recur(0)

    def grayCodeFailed(self, n: int) -> List[int]:
        failed = set()

        def backtrack(node, end, path, seen, mask):
            if len(path) > (1 << n): return []
            if node == end:
                if len(path) + 1 == (1 << n): return path + [node]
                return []
            new_mask = mask | (1 << node)
            if (node, new_mask) in failed: return []
            seen.add(node)
            path.append(node)
            for i in range(n):
                nei = node - (1 << i)
                if nei >= 0 and nei not in seen:
                    r = backtrack(nei, end, path, seen, new_mask)
                    if len(r) == (1 << n): return r
                nei = node + (1 << i)
                if nei < (1 << n) and nei not in seen:
                    r = backtrack(nei, end, path, seen, new_mask)
                    if len(r) == (1 << n): return r
            path.pop()
            seen.remove(node)
            failed.add((node, new_mask))
            return []

        for i in range(n):
            nei = 1 << i
            r = backtrack(0, nei, [], set(), 0)
            if len(r) == (1 << n): return r
        return []

    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        dummy_less_head = ListNode(None)
        dummy_less_tail = dummy_less_head
        dummy_greater_head = ListNode(None)
        dummy_greater_tail = dummy_greater_head

        node = head
        while node is not None:
            if node.val >= x:
                dummy_greater_tail.next = node
                dummy_greater_tail = node
            else:
                dummy_less_tail.next = node
                dummy_less_tail = node
            node = node.next
        dummy_less_tail.next = None
        dummy_greater_tail.next = None

        if dummy_less_tail != dummy_less_head:
            dummy_less_tail.next = dummy_greater_head.next
            return dummy_less_head.next
        else:
            return dummy_greater_head.next

    def numDecodings(self, s: str) -> int:
        dp = [0] * len(s)
        dp.append(1)
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '0': continue
            dp[i] = dp[i + 1]
            if i + 1 < len(s):
                x = int(s[i:i + 2])
                if 10 <= x <= 26:
                    dp[i] += dp[i + 2]
        return dp[0]

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy_head = ListNode(None)
        dummy_head.next = head
        prev = dummy_head
        node = head
        reverse_tail = head
        for i in range(1, right + 1):
            if i < left:
                prev = node
            if i == right:
                reverse_tail = node
            node = node.next
        lower_half = reverse_tail.next
        reverse_tail.next = None
        reverse_head = prev.next
        prev.next = None

        def reverse(head):
            new_tail = head
            new_head = head
            while new_tail.next is not None:
                tmp = new_tail.next
                new_tail.next = new_tail.next.next
                tmp.next = new_head
                new_head = tmp
            new_tail.next = None
            return new_head, new_tail

        reverse_head, reverse_tail = reverse(reverse_head)
        reverse_tail.next = lower_half
        prev.next = reverse_head
        return dummy_head.next

    def restoreIpAddresses(self, s: str) -> List[str]:
        memo = dict()

        def recur(i, digit):
            if (i, digit) in memo:
                return memo[i, digit]
            if i >= len(s): return []
            if digit == 1:
                if s[i] == '0':
                    if i != len(s) - 1: return []
                    else: return ['0']
                else:
                    if int(s[i:]) <= 255:
                        return [s[i:]]
                    else: return []

            results = []
            if s[i] == '0':
                sub_results = recur(i + 1, digit - 1)
                for r in sub_results:
                    results.append('0.' + r)
            else:
                for j in range(i, i + 3):
                    if j >= len(s): break
                    x = int(s[i:j + 1])
                    if x > 255: break
                    sub_results = recur(j + 1, digit - 1)
                    for r in sub_results:
                        results.append(s[i:j + 1] + '.' + r)
            memo[i, digit] = results
            return results
        return recur(0, 4)

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        dp = dict()

        def recur(left, right):
            if (left, right) in dp: return dp[left, right]
            if left == right:
                return [TreeNode(left)]

            result = []
            for i in range(left, right + 1):
                if i != left:
                    left_node_list = recur(left, i - 1)
                else:
                    left_node_list = [None]
                if i != right:
                    right_node_list = recur(i + 1, right)
                else:
                    right_node_list = [None]

                for l in left_node_list:
                    for r in right_node_list:
                        node = TreeNode(i)
                        node.left = l
                        node.right = r
                        result.append(node)
            dp[left, right] = result
            return result
        return recur(1, n)


    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        1, 6, 3, 4, 5, 2, 7
        Do not return anything, modify root in-place instead.
        """
        self.first_node = None
        self.second_node = None
        self.prev_biggest = None

        def recur(node):
            if node.left is not None:
                recur(node.left)

            if self.prev_biggest is not None:
                if self.prev_biggest.val > node.val:
                    self.first_node = self.prev_biggest
                    self.second_node = node
                else:
                    self.prev_biggest = node
            else:
                self.prev_biggest = node

            if node.right is not None:
                recur(node.right)
        recur(root)
        self.first_node.val, self.second_node.val = self.second_node.val, self.first_node.val

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        if root is None: return result
        def expand(level):
            while len(result) <= level:
                result.append([])

        def recur(node, level):
            expand(level)
            result[level].append(node.val)
            if node.left is not None:
                recur(node.left, level + 1)
            if node.right is not None:
                recur(node.right, level + 1)
        recur(root, 0)
        for i in range(len(result)):
            if i % 2 == 1:
                result[i] = list(reversed(result[i]))
        return result

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        reverse_map = {n: i for i, n in enumerate(inorder)}
        def recur(left, right):
            if left > right: return None
            node = TreeNode(postorder.pop())
            mid = reverse_map[node.val]
            node.right = recur(mid + 1, right)
            node.left = recur(left, mid - 1)
            return node
        return recur(0, len(inorder) - 1)

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        if root is None: return result
        def expand(level):
            while len(result) <= level:
                result.append([])

        def recur(node, level):
            expand(level)
            result[level].append(node.val)
            if node.left is not None:
                recur(node.left, level + 1)
            if node.right is not None:
                recur(node.right, level + 1)

        recur(root, 0)
        return list(reversed(result))

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        values = []
        while head is not None:
            values.append(head.val)
            head = head.next

        def recur(left, right):
            if left > right: return None
            mid = (left + right) // 2
            node = TreeNode(values[mid])
            node.left = recur(left, mid - 1)
            node.right = recur(mid + 1, right)
            return node
        return recur(0, len(values) - 1)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None: return []
        result = []
        def backtrack(node, paths, total):
            if node.left is None and node.right is None:
                if total + node.val == targetSum:
                    result.append(paths + [node.val])
                return
            paths.append(node.val)
            if node.left is not None:
                backtrack(node.left, paths, total + node.val)
            if node.right is not None:
                backtrack(node.right, paths, total + node.val)
            paths.pop()
        backtrack(root, [], 0)
        return result

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None: return
        def recur(node):
            # return first node and last node
            if node.left is None and node.right is None:
                return node, node

            if node.right is None:
                left_head, left_tail = recur(node.left)
                node.right = left_head
                node.left = None
                return node, left_tail
            if node.left is None:
                _, right_tail = recur(node.right)
                return node, right_tail

            left_head, left_tail = recur(node.left)
            right_head, right_tail = recur(node.right)
            node.left = None
            node.right = left_head
            left_tail.right = right_head
            return node, right_tail
        recur(root)

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root or root.left is None: return root

        # def recur(node):
        #     if node.left is None: return
        #     node.left.next = node.right
        #     recur(node.left)
        #     recur(node.right)
        #
        # recur(root)
        def link_middle(left_node, right_node):
            left_node.next = right_node
            if left_node.left is None: return
            link_middle(left_node.left, left_node.right)
            link_middle(right_node.left, right_node.right)
            link_middle(left_node.right, right_node.left)
        link_middle(root.left, root.right)
        return root

    def connect(self, root: 'Node') -> 'Node':
        head = root
        while head is not None:
            new_head = None
            prev = None
            while head is not None:
                if head.left is not None:
                    if new_head is None: new_head = head.left
                    if prev is not None:
                        prev.next = head.left
                    prev = head.left

                if head.right is not None:
                    if new_head is None: new_head = head.right
                    if prev is not None:
                        prev.next = head.right
                    prev = head.right
                head = head.next
            head = new_head
        return root

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        import math
        dp = []
        for t in triangle:
            dp.append([math.inf] * len(t))

        for i in range(len(triangle) - 1, -1, -1):
            for j in range(len(triangle[i])):
                if i == len(triangle) - 1:
                    dp[i][j] = triangle[i][j]
                else:
                    dp[i][j] = triangle[i][j] + dp[i + 1][j]
                    if j + 1 < len(triangle[i + 1]):
                        dp[i][j] = min(dp[i][j], triangle[i][j] + dp[i + 1][j + 1])
        return dp[0][0]

    def maxProfit(self, prices: List[int]) -> int:
        # 1 2 3 4 5 -> 1 5
        # 5 1 2 3 4 6 -> 5 1 6
        total = 0
        prev = prices[0]
        for i in range(1, len(prices)):
            if prices[i] > prev:
                total += prices[i] - prev
                prev = prices[i]
            else:
                prev = prices[i]
        return total

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        total = 0
        stack = [(root, 0)]
        while stack:
            node, running_total = stack.pop()
            if node.left is None and node.right is None:
                total += running_total * 10 + node.val
            if node.left is not None:
                stack.append((node.left, running_total * 10 + node.val))
            if node.right is not None:
                stack.append((node.right, running_total * 10 + node.val))
        return total

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        M = len(board)
        N = len(board[0])
        seen = [[False for _ in range(N)] for _ in range(M)]

        def gen(i, j):
            dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            for di, dj in dirs:
                if i + di < 0 or i + di >= M: continue
                if j + dj < 0 or j + dj >= N: continue
                yield i + di, j + dj

        def dfs(i, j):
            seen[i][j] = True
            for ni, nj in gen(i, j):
                if board[ni][nj] == 'X': continue
                if seen[ni][nj]: continue
                dfs(ni, nj)

        for i in range(M):
            if board[i][0] == '0' and not seen[i][0]:
                dfs(i, 0)
            if board[i][N - 1] == '0' and not seen[i][N - 1]:
                dfs(i, N - 1)

        for j in range(N):
            if board[0][j] == '0' and not seen[0][j]:
                dfs(0, j)
            if board[M - 1][j] == '0' and not seen[M - 1][j]:
                dfs(M - 1, j)

        for i in range(M):
            for j in range(N):
                if not seen[i][j] and board[i][j] == '0':
                    board[i][j] = 'X'

    def partition(self, s: str) -> List[List[str]]:
        dp = dict()

        def recur(i):
            if i in dp: return dp[i]
            if i >= len(s): return [[]]
            result = []
            for j in range(i, len(s)):
                sub_str = s[i:j + 1]
                if sub_str == sub_str[::-1]:
                    xs = recur(j + 1)
                    for x in xs:
                        result.append([sub_str] + x)
            dp[i] = result
            return result
        return recur(0)

    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node: return node

        d = dict()

        def build(node):
            if node.val in d: return d[node.val]
            new_node = Node(node.val, [])
            d[node.val] = new_node
            for nei in node.neighbors:
                new_node.neighbors.append(build(nei))
            return new_node
        return build(node)

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost): return -1
        tank = 0
        start = 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1
                tank = 0
        return start

    """
    # Definition for a Node.
    class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random
    """
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        d = dict()
        node = head
        while node is not None:
            d[node] = Node(node.val)
            node = node.next

        node = head
        while node is not None:
            new_node = d[node]
            if node.next is not None:
                new_node.next = d[node.next]
            if node.random is not None:
                new_node.random = d[node.random]
            node = node.next
        return d[head]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = dict()

        trie = dict()
        END = 'END'
        for w in wordDict:
            t = trie
            for c in w:
                if c not in t:
                    t[c] = dict()
                t = t[c]
            t[END] = None

        def recur(i):
            if i >= len(s): return True
            if i in dp: return dp[i]
            t = trie
            matched = False
            for j in range(i, len(s)):
                if s[j] not in t: break
                t = t[s[j]]
                if END in t:
                    if recur(j + 1):
                        matched = True
                        break
            dp[i] = matched
            return matched
        return recur(0)

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # N steps -> X + C*Y
        # 2X + 2C*Y

        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast: break
        if not fast or not fast.next: return None

        while head != slow:
            head, slow = head.next, slow.next
        return head

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        node = head
        count = 0
        while node is not None:
            count += 1
            node = node.next
        if count <= 1: return head
        N = (count + 1) // 2

        node = head
        while N > 1:
            node = node.next
            N -= 1

        second_head = node.next
        node.next = None

        def reverse(node):
            head = None
            while node is not None:
                tmp = node.next
                node.next = head
                head = node
                node = tmp
            return head

        second_head = reverse(second_head)
        prev = head
        first_pointer = head.next
        second_pointer = second_head
        while first_pointer or second_pointer:
            prev.next = second_pointer
            second_pointer = second_pointer.next
            prev.next.next = first_pointer
            prev = first_pointer
            if first_pointer is not None:
                first_pointer = first_pointer.next

    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for t in tokens:
            if t == '*':
                n1, n2 = stack.pop(), stack.pop()
                stack.append(n1 * n2)
            elif t == '-':
                n1 = stack.pop()
                n2 = stack.pop()
                stack.append(n2 - n1)
            elif t == '+':
                n1, n2 = stack.pop(), stack.pop()
                stack.append(n2 + n1)
            elif t == '/':
                n1 = stack.pop()
                n2 = stack.pop()
                stack.append(int(float(n2)/n1))
            else:
                stack.append(int(t))
        return stack[0]

    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 0
        max_pre = nums[0]
        min_pre = nums[0]
        max_so_far = nums[0]

        for i in range(1, len(nums)):
            n = nums[i]
            max_current = max(max_pre * n, min_pre * n, n)
            min_current = min(min_pre * n, max_pre * n, n)
            max_so_far = max(max_current, max_so_far)
            max_pre, min_pre = max_current, min_current
        return max_so_far

    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        import math
        dummy_head = ListNode(-math.inf)

        def insert(node):
            prev = dummy_head
            while prev.next is not None:
                if prev.next.val >= node.val: break
                prev = prev.next
            node.next = prev.next
            prev.next = node

        node = head
        while node is not None:
            tmp = node
            node = node.next
            insert(tmp)
        return dummy_head.next

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        count = 0
        node = head
        while node is not None:
            count += 1
            node = node.next
        return self.sortListHelper(head, count)

    def sortListHelper(self, head, count):
        if head is None or head.next is None: return head
        c = count // 2
        node = head
        while c - 1 > 0:
            node = node.next
            c -= 1

        tmp = node.next
        node.next = None
        left_head = self.sortListHelper(head, count // 2)
        right_head = self.sortListHelper(tmp, count - count // 2)
        return self.sortListHelperMerge(left_head, right_head)

    def sortListHelperMerge(self, l1, l2):
        if l1 is None: return l2
        if l2 is None: return l1
        dummy_head = ListNode(None)
        tail = dummy_head

        while l1 is not None or l2 is not None:
            if l1 is not None and l2 is not None:
                if l1.val < l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
            elif l1 is not None:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
            tail.next = None
        return dummy_head.next

    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            if nums[left] <= nums[right]: return nums[left]
            mid = (left + right) // 2
            if nums[mid] >= nums[left]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        def recur(node):
            # return new root and new parent
            if node.left is None:
                return node, node
            new_root, new_parent = recur(node.left)
            new_parent.left = node.right
            new_parent.right = node
            node.left = None
            node.right = None
            return new_root, node
        return recur(root)[0]

    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        from collections import Counter
        counter = Counter()

        max_length = 0
        left = 0
        right = 0
        while right < len(s):
            counter[s[right]] += 1
            while len(counter) > 2:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
            max_length = max(max_length, right - left + 1)
            right += 1
        return max_length

    def isOneEditDistance(self, s: str, t: str) -> bool:
        if s == t: return False
        if abs(len(s) - len(t)) > 1: return False
        if len(s) < len(t): return self.isOneEditDistance(t, s)

        i = 0
        while i < len(s):
            if i >= len(t): break
            if s[i] != t[i]: break
            i += 1
        return s[i + 1:] == t[i + 1:] or s[i + 1:] == t[i:]

    def findPeakElement(self, nums: List[int]) -> int:
        import math
        def get(i):
            if i < 0 or i >= len(nums): return -math.inf
            return nums[i]

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_minus_1 = get(mid - 1)
            mid_val = get(mid)
            mid_plus_1 = get(mid + 1)
            if mid_minus_1 < mid_val > mid_plus_1:
                return mid
            elif mid_minus_1 < mid_val < mid_plus_1:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def compareVersion(self, version1: str, version2: str) -> int:
        ds1 = version1.split('.')
        ds2 = version2.split('.')

        def compareInt(i1, i2):
            if i1 < i2: return -1
            elif i1 > i2: return 1
            else: return 0

        if len(ds1) < len(ds2):
            ds1 = ds1 + ['0'] * (len(ds2) - len(ds1))
        else:
            ds2 = ds2 + ['0'] * (len(ds1) - len(ds2))

        i = 0
        while i < len(ds1) and i < len(ds2):
            x = compareInt(int(ds1[i]), int(ds2[i]))
            if x != 0: return x
            i += 1
        return 0

    def maximumGap(self, nums: List[int]) -> int:
        min_val, max_val = min(nums), max(nums)
        if len(nums) < 2 or min_val == max_val: return 0
        buckets = [[] for _ in range(len(nums) + 1)]
        # 1 3 5 7 -> [0, bucket_size) .... [bucket_size * (n - 1), bucket_size * n)
        bucket_size = (max_val - min_val) / (len(buckets) - 1)

        for n in nums:
            idx = int((n - min_val) / bucket_size)
            buckets[idx].append(n)

        global_max = 0
        prev_max = None
        for i in range(len(nums) + 1):
            if not buckets[i]: continue
            min_current, max_current = min(buckets[i]), max(buckets[i])
            if prev_max is not None:
                global_max = max(global_max, min_current - prev_max)
            prev_max = max_current
        return global_max

    def calculate(self, s: str) -> int:

        def recur(i):
            result = 0
            sign = 1
            buffer = []
            while i < len(s):
                if s[i] == ')':
                    if buffer:
                        result += sign * int(''.join(buffer))
                    return result, i + 1
                elif s[i] == '+':
                    if buffer:
                        result += sign * int(''.join(buffer))
                    sign = 1
                    buffer = []
                    i += 1
                elif s[i] == '-':
                    if buffer:
                        result += sign * int(''.join(buffer))
                    sign = -1
                    buffer = []
                    i += 1
                elif s[i] == ' ':
                    i += 1
                elif s[i] == '(':
                    r, i = recur(i + 1)
                    result += sign * r
                    sign = 1
                else:
                    buffer.append(s[i])
                    i += 1
            if buffer:
                result += sign * int(''.join(buffer))
            return result, len(s)
        return recur(0)[0]

    def countPrimes(self, n: int) -> int:
        if n < 3: return 0
        is_prime = [True] * n
        is_prime[0], is_prime[1] = False, False

        for i in range(2, int(n**0.5)+1):
            if not is_prime[i]: continue
            j = 2
            while i * j < n:
                is_prime[i * j] = False
                j += 1
        return sum(is_prime)

    def closestPrimes(self, left: int, right: int) -> List[int]:
        import math
        right += 1
        is_prime = [True] * right
        is_prime[0], is_prime[1] = False, False

        for i in range(2, int(right**0.5)+1):
            if not is_prime[i]: continue
            j = 2
            while i * j < right:
                is_prime[i * j] = False
                j += 1

        primes = [i for i in range(left, right) if is_prime[i]]
        if len(primes) <= 1: return [-1, -1]

        min_diff = math.inf
        low_idx = 0
        for i in range(1, len(primes)):
            if min_diff > primes[i] - primes[i - 1]:
                low_idx = i - 1
                min_diff =primes[i] - primes[i - 1]
        return [primes[low_idx], primes[low_idx + 1]]

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            x = numbers[left] + numbers[right]
            if x == target: return [left + 1, right + 1]
            if x < target:
                left += 1
            else:
                right -= 1

    def trailingZeroes(self, n: int) -> int:
        def count(x):
            i = 1
            c = 0
            while x ** i <= n:
                c += n // (x ** i)
                i += 1
            return c
        return min(count(2), count(5))

    def largestNumber(self, nums: List[int]) -> str:
        from functools import cmp_to_key
        def compare(x, y):
            return -int(x + y) + int(y + x)


        nums = [str(n) for n in nums]
        nums.sort(key=cmp_to_key(compare))
        return str(int(''.join(nums)))

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        if len(s) <= 10: return []
        from collections import Counter
        counter = Counter()
        for i in range(9, len(s)):
            key = s[i - 9:i + 1]
            counter[key] += 1
        return [k for k, v in counter.items() if v > 1]

    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverse(low, high):
            left, right = low, high
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        k = k % len(nums)
        # [4 3 2 1 7 6 5]
        reverse(0, len(nums) - k - 1)
        reverse(len(nums) - k, len(nums) - 1)
        reverse(0, len(nums) - 1)

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None: return []
        levels = []

        def expand(level):
            while len(levels) <= level:
                levels.append(None)

        stack = [(root, 0)]

        while stack:
            node, level = stack.pop()
            expand(level)
            levels[level] = node.val
            if node.right is not None:
                stack.append((node.right, level + 1))
            if node.left is not None:
                stack.append((node.left, level + 1))
        return levels

    def minSubArrayLenBS(self, target: int, nums: List[int]) -> int:
        def check(length):
            left, right = 0, 0
            s = 0
            while right < len(nums):
                s += nums[right]
                if right - left + 1 > length:
                    s -= nums[left]
                    left += 1
                if right - left + 1 == length:
                    if s >= target: return True
                right += 1
            return False

        left, right = 1, len(nums)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        if left == len(nums) + 1: return 0
        return left

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left, right = 0, 0
        s = 0
        smallest_length = len(nums) + 1
        while right < len(nums):
            s += nums[right]
            while s >= target:
                smallest_length = min(smallest_length, right - left + 1)
                s -= nums[left]
                left += 1
            right += 1
        if smallest_length == len(nums) + 1: return 0
        return smallest_length

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        from collections import deque
        graph = [[] for _ in range(numCourses)]
        degrees = [0] * numCourses
        for p in prerequisites:
            graph[p[1]].append(p[0])
            degrees[p[0]] += 1

        zero_degrees = deque()
        for i in range(numCourses):
            if degrees[i] != 0: continue
            zero_degrees.append(i)

        order = []
        while zero_degrees:
            node = zero_degrees.popleft()
            order.append(node)
            for nei in graph[node]:
                degrees[nei] -= 1
                if degrees[nei] == 0:
                    zero_degrees.append(nei)
        if len(order) == numCourses: return order
        return []

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        M = len(matrix)
        N = len(matrix[0])
        ones_to_right = [[0 for _ in range(N)] for _ in range(M)]
        ones_to_low = [[0 for _ in range(N)] for _ in range(M)]

        for i in range(M):
            to_right = 0
            for j in range(N - 1, -1, -1):
                if matrix[i][j] == '0':
                    to_right = 0
                else:
                    to_right += 1
                    ones_to_right[i][j] = to_right
        for j in range(N):
            to_low = 0
            for i in range(M - 1, -1, -1):
                if matrix[i][j] == '0':
                    to_low = 0
                else:
                    to_low += 1
                    ones_to_low[i][j] = to_low

        dp = [[0 for _ in range(N)] for _ in range(M)]
        max_length_size = 0
        for i in range(M - 1, -1, -1):
            for j in range(N - 1, -1, -1):
                if matrix[i][j] == '0': continue
                if i + 1 < M and j + 1 < N:
                    max_sub = dp[i + 1][j + 1]
                    max_low = ones_to_low[i][j]
                    max_right = ones_to_right[i][j]
                    length = min(max_sub + 1, max_low, max_right)
                    dp[i][j] = length
                else:
                    dp[i][j] = 1
                max_length_size = max(max_length_size, dp[i][j] ** 2)
        return max_length_size

    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        graph = [[] for _ in range(n)]
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        seen = [False] * n

        def dfs(node):
            for nei in graph[node]:
                if seen[nei]: continue
                seen[nei] = True
                dfs(nei)

        count = 0
        for i in range(n):
            if not seen[i]:
                count += 1
                seen[i] = True
                dfs(i)
        return count

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        spots = []
        for inter in intervals:
            spots.append((inter[0], 1))
            spots.append((inter[1], -1))
        spots.sort()
        count = 0
        max_count = 0
        for s in spots:
            count += s[1]
            max_count = max(max_count, count)
        return max_count

    def majorityElement_closeButFailed(self, nums: List[int]) -> List[int]:
        if not nums: return []
        if len(nums) == 1: return nums
        major1 = nums[0]
        count1 = 1
        major2 = nums[1]
        count2 = 1
        for i in range(2, len(nums)):
            if nums[i] == major1:
                count1 += 1
            elif nums[i] == major2:
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1
                if count1 < 0:
                    major1 = nums[i]
                    count1 = 1
                    count2 += 1
                elif count2 < 0:
                    major2 = nums[i]
                    count2 = 1
                    count1 += 1
        result = []
        if count1 > 0: result.append(major1)
        if count2 > 0: result.append(major2)
        return result

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        def recur(node, prev_count):
            left_count = 0
            if node.left is not None:
                left_count, val = recur(node.left, prev_count)
                if val is not None: return 0, val
            if left_count + prev_count + 1 == k: return 0, node.val
            right_count = 0
            if node.right is not None:
                right_count, val = recur(node.right, prev_count + left_count + 1)
                if val is not None: return 0, val
            return left_count + right_count + 1, None
        return recur(root, 0)[1]

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def recur(node, target, path):
            path.append(node)
            if node == target:
                return path, True
            if node.left is not None:
                p, result = recur(node.left, target, path)
                if result: return p, True
            if node.right is not None:
                p, result = recur(node.right, target, path)
                if result: return p, True
            path.pop()
            return None, False

        p_path = recur(root, p, [])[0]
        q_path = recur(root, q, [])[0]
        i = 0
        while i < len(p_path) and i < len(q_path):
            if p_path[i] != q_path[i]:
                return p_path[i - 1]
            i += 1
        if i < len(p_path): return p_path[i - 1]
        return q_path[i - 1]

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        product_left = []
        for n in nums:
            if not product_left:
                product_left.append(n)
            else:
                product_left.append(n * product_left[-1])
        product_right = [0] * len(nums)
        product_right[-1] = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            product_right[i] = nums[i] * product_right[i + 1]

        result = []
        for i in range(len(nums)):
            if i == 0:
                result.append(product_right[i + 1])
            elif i == len(nums) - 1:
                result.append(product_left[i - 1])
            else:
                result.append(product_left[i - 1] * product_right[i + 1])
        return result

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        M = len(matrix)
        N = len(matrix[0])
        i, j = 0, N - 1
        while i < M and j >= 0:
            if matrix[i][j] == target: return True
            if matrix[i][j] < target:
                i += 1
            else:
                j -= 1
        return False

    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        pos1 = []
        pos2 = []
        for i, w in enumerate(wordsDict):
            if w == word1:
                pos1.append(i)
            if w == word2:
                pos2.append(i)

        min_distance = len(wordsDict)
        ptr1 = 0
        ptr2 = 0
        while ptr1 < len(pos1) and ptr2 < len(pos2):
            x = abs(pos1[ptr1] - pos2[ptr2])
            if x != 0:
                min_distance = min(min_distance, x)
            if pos1[ptr1] < pos2[ptr2]:
                ptr1 += 1
            else:
                ptr2 += 1
        return min_distance

    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        def convert(s):
            digits = [ord(c) for c in s]
            min_digits = min(digits)
            digits = [d - min_digits for d in digits]
            base = 26 - digits[0]
            digits = [str((d + base) % 26) for d in digits]
            return ','.join(digits)
        d = dict()
        for s in strings:
            k = convert(s)
            if k not in d:
                d[k] = [s]
            else:
                d[k].append(s)
        return [v for v in d.values()]

    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        counter = 0
        def recur(node):
            nonlocal counter
            # return if they all have same value, if yes, which value
            is_left_same, left_val = True, node.val
            if node.left is not None:
                is_left_same, left_val = recur(node.left)
            is_right_same, right_val = True, node.val
            if node.right is not None:
                is_right_same, right_val = recur(node.right)
            if is_left_same and is_right_same and left_val == right_val == node.val:
                counter += 1
                return True, node.val
            return False, None
        if not root: return 0
        recur(root)
        return counter

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        graph = [[] for _ in range(n)]
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        seen = [False] * n
        seen[0] = True
        stack = [(0, None)]
        while stack:
            node, parent = stack.pop()
            for nei in graph[node]:
                if nei == parent: continue
                if seen[nei]: return False
                seen[nei] = True
                stack.append((nei, node))
        return seen == [True] * n

    def generatePalindromes(self, s: str) -> List[str]:
        from collections import Counter
        counter = Counter(s)
        one_counts = 0
        for v in counter.values():
            if v % 2 == 1: one_counts += 1
        if one_counts > 1: return []

        key_v = [[k, v] for k, v in counter.items()]

        dp = dict()
        def get_key(key_v):
            return ','.join([str(l[0]) + ':' + str(l[1]) for l in key_v])

        def recur(key_v, length):
            key = get_key(key_v)
            if key in dp:
                return dp[key]
            if length == 0: return ['']
            result = []
            for kv in key_v:
                if kv[1] == 0: continue
                if kv[1] == 1:
                    if length == 1:
                        result.append(kv[0])
                else:
                    kv[1] -= 2
                    sub_result = recur(key_v, length - 2)
                    for s in sub_result:
                        result.append(kv[0] + s + kv[0])
                    kv[1] += 2
            dp[key] = result
            return result
        return recur(key_v, len(s))


    def majorityElement(self, nums):
        import math
        if len(nums) <= 1: return nums
        count1, count2, cand1, cand2 = 0, 0, -math.inf, math.inf
        for n in nums:
            if n == cand1:
                count1 += 1
            elif n == cand2:
                count2 += 1
            elif count1 == 0:
                count1 = 1
                cand1 = n
            elif count2 == 0:
                count2 = 1
                cand2 = n
            else:
                count1 -= 1
                count2 -= 1
        result = []
        if nums.count(cand1) > len(nums) // 3: result.append(cand1)
        if nums.count(cand2) > len(nums) // 3: result.append(cand2)
        return result

    def verifyPreorder(self, preorder: List[int]) -> bool:
        import math
        if not preorder: return True

        def build_tree(i, left_bound, right_bound):
            if i == len(preorder): return i
            if preorder[i] <= left_bound or preorder[i] >= right_bound: return i
            og_i = i
            i = build_tree(i + 1, left_bound, preorder[og_i])
            return build_tree(i, preorder[og_i], right_bound)
        return build_tree(0, -math.inf, math.inf) == len(preorder)

    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        result = 0
        for i in range(len(nums)):
            j, k = i + 1, len(nums) - 1
            t = target - nums[i]
            while j < k:
                if nums[j] + nums[k] < t:
                    result += k - j
                    j += 1
                else:
                    k -= 1
        return result

    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        result = []
        for s in strs:
            result.append(s.replace('#', '##'))
        return ' # '.join(result)


    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        arrays = s.split(' # ')
        result = []
        for arr in arrays:
            result.append(arr.replace('##', '#'))
        return result

    def diffWaysToCompute(self, expression: str) -> List[int]:
        seen = dict()
        ops = {'+', '-', '*'}

        def recur(s):
            if s in seen: return seen[s]
            contains_marks = False
            result = []
            for i, c in enumerate(s):
                if c not in ops: continue
                contains_marks = True
                l = recur(s[:i])
                r = recur(s[i + 1:])
                for x in l:
                    for y in r:
                        if c == '+':
                            result.append(x + y)
                        elif c == '-':
                            result.append(x - y)
                        else:
                            result.append(x * y)
            if not contains_marks:
                result.append(int(s))
            seen[s] = result
            return result
        return recur(expression)

    def singleNumber(self, nums: List[int]) -> List[int]:
        def get_xor(nums):
            x = nums[0]
            for i in range(1, len(nums)):
                x = x^nums[i]
            return x
        x = get_xor(nums)
        bit = 1
        while bit & x == 0:
            bit = bit << 1

        group_unset = []
        group_set = []
        for n in nums:
            if n & bit == 0:
                group_unset.append(n)
            else:
                group_set.append(n)
        return [
            get_xor(group_set),
            get_xor(group_unset)
        ]

    def hIndex(self, citations: List[int]) -> int:
        import math
        left, right = 0, len(citations)
        while left <= right:
            mid = math.ceil((left + right) / 2)
            if mid == 0:
                return 0
            if citations[len(citations) - mid] >= mid:
                left = mid + 1
            else:
                right = mid - 1
        return right

    def numWays(self, n: int, k: int) -> int:
        if n == 1: return k
        dp = [[1, 0] for _ in range(k)]
        for i in range(n - 1):
            total = sum(d[0] + d[1] for d in dp)
            new_dp = [[0, 0] for _ in range(k)]
            for j in range(k):
                new_dp[j][0] = total - dp[j][0] - dp[j][1]
                new_dp[j][1] = dp[j][0]
            dp = new_dp
        return sum(d[0] + d[1] for d in dp)

    def findCelebrity(self, n: int) -> int:
        def knows(a: int, b: int) -> bool:
            pass

        def check(candidate):
            for i in range(n):
                if i == candidate: continue
                if knows(candidate, i): return False
                if not knows(i, candidate): return False
            return True

        candidate = 0
        for i in range(1, n):
            if knows(candidate, i):
                candidate = i
        if check(candidate): return candidate

        return -1

    def numSquaresTopDown(self, n: int) -> int:
        import math
        squares = []
        for i in range(1, n + 1):
            if i ** 2 > n: break
            squares.append(i ** 2)

        seen = dict()
        def recur(x):
            if x in seen: return seen[x]
            if x == 0: return 0

            result = math.inf
            for n in squares:
                if n > x: break
                result = min(result, 1 + recur(x - n))
            seen[x] = result
            return result
        return recur(n)

    def numSquares(self, n: int) -> int:
        import math
        squares = []
        for i in range(1, n + 1):
            if i ** 2 > n: break
            squares.append(i ** 2)

        dp = [math.inf] * (1 + n)
        dp[0] = 0
        for i in range(n):
            for s in squares:
                if i + s > n: break
                dp[i + s] = min(dp[i + s], 1 + dp[i])
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        dp = [0, 0]
        for n in nums:
            new_dp = [0, 0]
            new_dp[0] = dp[1] + n
            new_dp[1] = max(dp)
            dp = new_dp
        return max(dp)

    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        def rob1(init, nums: List[int]) -> int:
            dp = init
            for n in nums:
                new_dp = [0, 0]
                new_dp[0] = dp[1] + n
                new_dp[1] = max(dp)
                dp = new_dp
            return max(dp)

        return max(rob1([nums[0], 0], nums[1:len(nums) - 1]),
                   rob1([0, 0], nums[1:]))

    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        parent = None
        node = root

        while node != p:
            if node.val > p.val:
                parent = node
                node = node.left
            else:
                node = node.right
        if node.right is None: return parent
        node = node.right
        while node.left is not None:
            node = node.left
        return node

    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        from collections import deque
        queue = deque()
        M = len(rooms)
        N = len(rooms[0])
        for i in range(M):
            for j in range(N):
                if rooms[i][j] == -1: continue
                elif rooms[i][j] == 0:
                    queue.append((i, j, 0))

        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        def gen(i, j):
            for di, dj in dirs:
                if i + di < 0 or i + di >= M: continue
                if j + dj < 0 or j + dj >= N: continue
                yield i + di, j + dj

        while queue:
            i, j, steps = queue.popleft()
            for ni, nj in gen(i, j):
                if rooms[ni][nj] == -1: continue
                if rooms[ni][nj] == 0: continue
                if rooms[ni][nj] <= steps + 1: continue
                rooms[ni][nj] = steps + 1
                queue.append((ni, nj, steps + 1))

    def findDuplicate(self, nums: List[int]) -> int:

        for i in range(len(nums)):
            if i == nums[i] - 1: continue
            if nums[i] == nums[nums[i] - 1]: return nums[i]

            while nums[i] != i + 1:
                left = nums[i]
                right = nums[nums[i] - 1]
                nums[i], nums[left - 1] = right, left
                if i != nums[i] - 1 and nums[i] == nums[nums[i] - 1]: return nums[i]

    def canWin(self, currentState: str) -> bool:
        seen = dict()
        def recur(s):
            if s in seen: return seen[s]
            if len(s) < 2: return False
            result = False
            for i in range(len(s) - 1):
                if s[i:i + 2] == '++':
                    new_s = s[:i] + '--' + s[i + 2:]
                    result = not recur(new_s)
                if result: break
            seen[s] = result
            return result
        return recur(currentState)

    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        global_max = 1
        def recur(node):
            nonlocal  global_max
            max_consecutive = 1
            if node.left is not None:
                left_max = recur(node.left)
                if node.val == node.left.val - 1:
                    max_consecutive = max(max_consecutive, 1 + left_max)
            if node.right is not None:
                right_max = recur(node.right)
                if node.val == node.right.val - 1:
                    max_consecutive = max(max_consecutive, 1 + right_max)
            global_max = max(global_max, max_consecutive)
            return max_consecutive
        recur(root)
        return global_max

    def getHint(self, secret: str, guess: str) -> str:
        cows = 0
        bulls = 0
        sd = defaultdict(set)
        gd = defaultdict(set)

        for i, c in enumerate(secret):
            sd[c].add(i)
        for i, c in enumerate(guess):
            gd[c].add(i)

        for key in sd.keys():
            if key not in gd: continue
            secret_set = sd[key]
            guess_set = gd[key]
            inter_set = secret_set.intersection(guess_set)
            bulls += len(inter_set)
            cows += min(len(secret_set) - len(inter_set), len(guess_set) - len(inter_set))
        return str(bulls) + 'A' + str(cows) + 'B'

    def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
        companies = [(set(c), i) for i, c in enumerate(favoriteCompanies)]
        companies.sort(key=lambda x: len(x[0]), reverse=True)
        result = []
        for i in range(len(companies)):
            found = False
            for j in range(len(companies)):
                if j == i: continue
                if len(companies[j][0]) < len(companies[i][0]): break
                if len(companies[j][0].intersection(companies[i][0])) == len(companies[i][0]):
                    found = True
                    break

            if not found: result.append(companies[i][1])
        result.sort()
        return result

    def coinChange(self, coins: List[int], amount: int) -> int:
        import math
        coins.sort()
        seen = dict()
        def recur(x):
            if x == 0: return 0
            if x in seen: return seen[x]
            min_result = math.inf
            for i in range(len(coins) - 1, -1, -1):
                if coins[i] > x: continue
                min_result = min(min_result, 1 + recur(x - coins[i]))
            seen[x] = min_result
            return min_result
        r = recur(amount)
        if r == math.inf: return -1
        return r

    def change(self, amount: int, coins: List[int]) -> int:
        dp = [[-1 for _ in range(amount + 1)] for _ in range(len(coins))]
        def recur(i, x):
            if x == 0: return 1
            if i >= len(coins): return 0
            if dp[i][x] != -1: return dp[i][x]
            result = 0
            for j in range(x // coins[i] + 1):
                result += recur(i + 1, x - j * coins[i])
            dp[i][x] = result
            return result
        return recur(0, amount)

    def maxProfit(self, prices: List[int]) -> int:
        import math
        hold = -math.inf
        not_hold_no_cd = 0
        not_hold_cd = -math.inf
        for p in prices:
            new_hold = max(hold, not_hold_no_cd - p)
            new_not_hold_no_cd = max(not_hold_no_cd, not_hold_cd)
            new_not_hold_cd = hold + p
            hold, not_hold_no_cd, not_hold_cd = new_hold, new_not_hold_no_cd, new_not_hold_cd
        return max(not_hold_cd, not_hold_no_cd)

    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n <= 2:
            return [i for i in range(n)]
        graph = [set() for _ in range(n)]
        for e in edges:
            graph[e[0]].add(e[1])
            graph[e[1]].add(e[0])
        remaining = n

        leaves = []
        for i in range(n):
            if len(graph[i]) == 1:
                leaves.append(i)

        while remaining > 2:
            remaining -= len(leaves)
            new_leaves = []
            for node in leaves:
                nei = graph[node].pop()
                graph[nei].remove(node)
                if len(graph[nei]) == 1:
                    new_leaves.append(nei)
            leaves = new_leaves
        return leaves

    def multiplySlow(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat1[0])):
                    result[i][j] += mat1[i][k] * mat2[k][j]
        return result

    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

        def compress(mat):
            rows = [[] for _ in range(len(mat))]
            cols = [[] for _ in range(len(mat[0]))]
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    if mat[i][j] == 0: continue
                    rows[i].append((j, mat[i][j]))
                    cols[j].append((i, mat[i][j]))
            return rows, cols

        rows1, _ = compress(mat1)
        _, cols2 = compress(mat2)
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                row, col = rows1[i], cols2[j]
                index_row, index_col = 0, 0
                while index_row < len(row) and index_col < len(col):
                    if row[index_row][0] == col[index_col][0]:
                        result[i][j] += row[index_row][1] * col[index_col][1]
                        index_col += 1
                        index_row += 1
                    elif row[index_row][0] > col[index_col][0]:
                        index_col += 1
                    else:
                        index_row += 1
        return result

    def removeDuplicateLetters(self, s: str) -> str:
        last_occur = dict()
        for i, c in enumerate(s):
            last_occur[c] = i
        stack = []
        for i, c in enumerate(s):
            if stack.count(c) != 0: continue
            while stack:
                if stack[-1] < c: break
                if last_occur[stack[-1]] > i:
                    stack.pop()
                else: break
            stack.append(c)
        return ''.join(stack)

    def maxProduct(self, words: List[str]) -> int:

        def get_bits(w):
            mask = 0
            for c in w:
                idx = ord(c) - ord('a')
                mask = mask | (1 << idx)
            return mask

        masks = [get_bits(w) for w in words]
        max_product = 0
        for i in range(1, len(words)):
            for j in range(i):
                if masks[i] & masks[j] != 0: continue
                max_product = max(max_product, len(words[i]) * len(words[j]))
        return max_product

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p: return not q
        stack = [(p, q)]
        while stack:
            node_p, node_q = stack.pop()
            if node_p is None and node_q is None: continue
            if not node_p or not node_q: return False
            if node_p.val != node_q.val: return False
            stack.append((node_p.left, node_q.left))
            stack.append((node_p.right, node_q.right))
        return True

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        stack = [(root.left, root.right)]
        while stack:
            left, right = stack.pop()
            if left is None and right is None: continue
            if not left or not right: return False
            if left.val != right.val: return False
            stack.append((left.left, right.right))
            stack.append((left.right, right.left))
        return True

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        max_depth = 0
        if not root: return max_depth
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))
        return max_depth

    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        max_length = 0
        prefix_sum_to_index = dict()
        prefix_sum_to_index[0] = -1
        running_sum = 0
        for i, n in enumerate(nums):
            running_sum += n
            if running_sum - k in prefix_sum_to_index:
                max_length = max(max_length, i - prefix_sum_to_index[running_sum - k])
            if running_sum not in prefix_sum_to_index:
                prefix_sum_to_index[running_sum] = i
        return max_length

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root: return True

        def recur(node):
            left_h = 0
            is_left_balanced = True
            if node.left is not None:
                left_h, is_left_balanced = recur(node.left)
            right_h = 0
            is_right_balanced = True
            if node.right is not None:
                right_h, is_right_balanced = recur(node.right)
            is_balanced = is_left_balanced and is_right_balanced and abs(left_h - right_h) <= 1
            return max(left_h, right_h) + 1, is_balanced
        return recur(root)[1]

    def findMin(self, nums: List[int]) -> int:
        return min(nums)

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        def length(node):
            l = 0
            while node is not None:
                l += 1
                node = node.next
            return l
        A_length = length(headA)
        B_length = length(headB)
        if A_length < B_length:
            headA, headB = headB, headA
            A_length, B_length = B_length, A_length

        for _ in range(A_length - B_length):
            headA = headA.next

        while headA != headB:
            headA = headA.next
            headB = headB.next
        return headA

    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        largest_tree = 1

        def recur(node):
            nonlocal largest_tree
            left_size = 0
            right_size = 0
            is_BST = True
            min_val, max_val = node.val, node.val
            if node.left is not None:
                left_size, is_left_BST, left_min, left_max = recur(node.left)
                is_BST = is_BST and is_left_BST and left_max < node.val
                min_val = left_min
            if node.right is not None:
                right_size, is_right_BST, right_min, right_max = recur(node.right)
                is_BST = is_BST and is_right_BST and right_min > node.val
                max_val = right_max
            if is_BST:
                largest_tree = max(largest_tree, left_size + right_size + 1)
            return left_size + right_size + 1, is_BST, min_val, max_val
        recur(root)
        return largest_tree

    def isValidSerialization(self, preorder: str) -> bool:
        preorder = preorder.split(',')
        def recur(i):
            if i >= len(preorder): return i, False
            if preorder[i] == '#':
                return i + 1, True
            start, is_valid = recur(i + 1)
            if not is_valid: return start, is_valid
            start, is_valid = recur(start)
            return start, is_valid
        start, is_valid = recur(0)
        return is_valid and start == len(preorder)

    def increasingTriplet(self, nums: List[int]) -> bool:
        import math
        has_smaller = [False] * len(nums)
        has_larger = [False] * len(nums)
        smaller = math.inf
        for i, n in enumerate(nums):
            if smaller < n: has_smaller[i] = True
            smaller = min(smaller, n)
        larger = -math.inf
        for i in range(len(nums) - 1, -1, -1):
            if larger > nums[i]: has_larger[i] = True
            larger = max(larger, nums[i])
        for i in range(1, len(nums) - 1):
            if has_larger[i] and has_smaller[i]: return True
        return False

    def rob(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        dp = dict()

        def recur(node, parent_robbed):
            if not node: return 0
            if (node, parent_robbed) in dp: return dp[node, parent_robbed]
            r = 0
            if not parent_robbed:
                r = max(r, node.val + recur(node.left, True) + recur(node.right, True))
            r = max(r, recur(node.left, False) + recur(node.right, False))
            dp[node, parent_robbed] = r
            return r
        return recur(root, False)

    def depthSum(self, nestedList: List[NestedInteger]) -> int:

        def recur(nestedList: List[NestedInteger], depth):
            r = 0
            for nl in nestedList:
                if nl.isInteger():
                    r += depth * nl.getInteger()
                else:
                    r += recur(nl.getList(), depth + 1)
            return r
        return recur(nestedList, 1)

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        from collections import Counter
        counter = Counter()
        max_length = 0
        left, right = 0, 0
        while right < len(s):
            counter[s[right]] += 1
            while len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
            max_length = max(max_length, right - left + 1)
            right += 1
        return max_length

    def integerBreak(self, n: int) -> int:
        memo = dict()
        memo[0] = 0
        memo[1] = 1
        memo[2] = 1
        memo[3] = 2

        def recur(x):
            if x in memo: return max(memo[x], x)
            r = x - 1
            for i in range(1, x):
                r = max(r, i * recur(x - i))
            memo[x] = r
            return r
        if n <= 3: return memo[n]
        return recur(n)

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        import heapq
        M = len(matrix)
        N = len(matrix[0])
        heap = []
        for i in range(M):
            heap.append((matrix[i][0], i, 1))
        heapq.heapify(heap)
        counter = 0
        last_x = 0
        while counter < k:
            last_x, row, next_col = heapq.heappop(heap)
            counter += 1
            if next_col < N:
                heapq.heappush(heap, (matrix[row][next_col], row, next_col + 1))
        return last_x

    def canMeasureWater(self, x: int, y: int, target: int) -> bool:
        if x + y < target: return False
        seen = set()
        def recur(x_water, y_water):
            if x_water + y_water == target: return True
            if (x_water, y_water) in seen: return False
            seen.add((x_water, y_water))
            r = recur(0, y_water)
            r = r or recur(x_water, 0)
            r = r or recur(x, y_water)
            r = r or recur(x_water, y)
            r = r or recur(min(x, x_water + y_water), max(0, y_water - (x - x_water)))
            r = r or recur(max(0, x_water - (y - y_water)), min(y, x_water + y_water))
            return r
        return recur(0, 0)

    def isReflected(self, points: List[List[int]]) -> bool:
        same_ys = defaultdict(set)
        for p in points:
            same_ys[p[1]].add(p[0])
        axis = None
        for k, v in same_ys.items():
            v = list(sorted(v))
            left, right = 0, len(v) - 1
            while left <= right:
                if axis is not None and axis != (v[left] + v[right]) / 2: return False
                axis = (v[left] + v[right]) / 2
                left += 1
                right -= 1
        return True


    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        if not nestedList: return 0
        flattened = []

        def recur(nestedList, depth):
            for ni in nestedList:
                if ni.isInteger():
                    flattened.append((ni.getInteger(), depth))
                else:
                    recur(ni.getList(), depth + 1)
        recur(nestedList, 1)
        max_depth = max(x[1] for x in flattened)
        total = 0
        for f in flattened:
            total += (max_depth - f[1] + 1) * f[0]
        return total

    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import Counter
        if not root: return []
        in_degrees = Counter()
        reverse_graph = dict()

        def recur(node, parent):
            reverse_graph[node] = parent
            in_degrees[parent] += 1
            if node.left is not None:
                recur(node.left, node)
            if node.right is not None:
                recur(node.right, node)
        recur(root, None)
        results = []
        last_results = []
        for n in reverse_graph.keys():
            if in_degrees[n] == 0: last_results.append(n)
        results.append(last_results)
        while last_results:
            current_results = []
            for n in last_results:
                p = reverse_graph[n]
                if p is None: continue
                in_degrees[p] -= 1
                if in_degrees[p] == 0:
                    current_results.append(p)
            if current_results:
                results.append(current_results)
            last_results = current_results
        results = [[n.val for n in l] for l in results]
        return results

    def plusOne(self, head: ListNode) -> ListNode:

        def recur(node):
            if node.next is not None:
                overflow = recur(node.next)
            else:
                overflow = 1

            node.val, overflow = (node.val + overflow) % 10,  (node.val + overflow) // 10
            return overflow
        overflow = recur(head)
        if overflow == 1:
            newHead = ListNode(overflow)
            newHead.next = head
            return newHead
        else: return head

    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        increments = [0] * (length + 1)
        for u in updates:
            increments[u[0]] += u[2]
            increments[u[1] + 1] -= u[2]
        result = [0] * length
        x = 0
        for i, incr in enumerate(increments):
            if i >= length: break
            x += incr
            result[i] = x
        return result

    def getSum(self, a: int, b: int) -> int:
        i = 0
        x = 1 << i
        overflow = 0
        result = 0
        while x <= a or x <= b:
            digit_a = 1 if (x & a) != 0 else 0
            digit_b = 1 if (x & b) != 0 else 0
            digit = (digit_a + digit_b + overflow) % 2
            overflow = (digit_a + digit_b + overflow) // 2
            result = result | (digit << i)
            i += 1
            x = x << 1
        result = result | (overflow<< i)
        return result


    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        import heapq
        # (sum, number, position in nums2)
        heap = []
        for i, n in enumerate(nums1):
            if i >= k: break
            heapq.heappush(heap, (nums2[0] + n, n, 0))
        result = []
        while len(result) < k:
            _, n, index = heapq.heappop(heap)
            result.append([n, nums2[index]])
            if index + 1 < len(nums2):
                heapq.heappush(heap, (nums2[index + 1] + n, n, index + 1))
        return result

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
        return points[:k]

    def minTotalDistanceTLE(self, grid: List[List[int]]) -> int:
        import math
        ones = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    ones.append((i, j))
        min_distance = math.inf
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                total_d = 0
                for one in ones:
                    total_d += abs(one[0] - i) + abs(one[1] - j)
                min_distance = min(min_distance, total_d)
        return min_distance

    def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
        if not root: return []
        from collections import deque
        smaller = deque()
        bigger = deque()

        def recur(node):
            if node.left is not None:
                recur(node.left)
            if node.val <= target:
                smaller.append(node.val)
            else:
                bigger.append(node.val)
            if node.right is not None:
                recur(node.right)
        recur(root)
        result = []
        while len(result) < k:
            if not smaller:
                result.append(bigger.popleft())
            elif not bigger:
                result.append(smaller.pop())
            else:
                if abs(bigger[0] - target) < abs(smaller[-1] - target):
                    result.append(bigger.popleft())
                else:
                    result.append(smaller.pop())
        return result

    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        from collections import deque
        bank = set(bank)
        if endGene not in bank: return -1
        if startGene == endGene: return 0
        seen = set()

        queue = deque()
        queue.append((startGene, 0))
        seen.add(startGene)
        while queue:
            node, steps = queue.popleft()
            for i in range(len(node)):
                for c in 'ACGT':
                    new_node = node[:i] + c + node[i + 1:]
                    if new_node in seen: continue
                    if new_node not in bank: continue
                    if new_node == endGene:
                        return steps + 1
                    seen.add(new_node)
                    queue.append((new_node, steps + 1))
        return -1


    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        from collections import Counter
        N = len(graph)
        groups = [[i, 1] for i in range(N)]
        def find(node):
            g = groups[node][0]
            if g != node:
                g = find(g)
            groups[node][0] = g
            return g

        def union(node1, node2):
            g1, g2 = find(node1), find(node2)
            if g1 == g2: return
            if groups[g1][1] > groups[g2][1]:
                groups[g2][0] = g1
                groups[g1][1] += groups[g2][1]
            else:
                groups[g1][0] = g2
                groups[g2][1] += groups[g1][1]

        for node in range(N):
            for nei in range(N):
                if nei == node: continue
                if graph[node][nei] == 1:
                    union(node, nei)

        malware_count = Counter()
        for i in initial:
            malware_count[find(i)] += 1
        group_size = []
        for i in initial:
            if malware_count[find(i)] > 1: group_size.append(0)
            else:
                group_size.append(groups[find(i)][1])


        max_node = 0
        for i in range(1, len(initial)):
            if group_size[i] > group_size[max_node]:
                max_node = i
            elif group_size[i] == group_size[max_node] and initial[max_node] > initial[i]:
                max_node = i
        return initial[max_node]

    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s % 2 != 0: return False
        target = s // 2
        memo = dict()
        def check(i, targetSum):
            if i == len(nums): return targetSum == 0
            if (i, targetSum) in memo: return memo[i, targetSum]
            if targetSum >= nums[i]:
                memo[i, targetSum] = (check(i + 1, targetSum) or check(i + 1, targetSum - nums[i]))
            else:
                memo[i, targetSum] = check(i + 1, targetSum)
            return memo[i, targetSum]
        return check(0, target)

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        N = len(heights)
        M = len(heights[0])
        reached = [[0 for _ in range(M)] for _ in range(N)]

        def gen(i, j):
            for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if i + di < 0 or i + di >= N: continue
                if j + dj < 0 or j + dj >= M: continue
                yield i + di, j + dj

        def propagate(initial_cells):
            seen = [[False for _ in range(M)] for _ in range(N)]
            for c in initial_cells:
                if seen[c[0]][c[1]]: continue
                seen[c[0]][c[1]] = True
                reached[c[0]][c[1]] += 1

            stack = initial_cells
            while stack:
                cell = stack.pop()
                for x, y in gen(cell[0], cell[1]):
                    if heights[x][y] < heights[cell[0]][cell[1]]: continue
                    if seen[x][y]: continue
                    seen[x][y] = True
                    reached[x][y] += 1
                    stack.append((x, y))
        pacific_cells = []
        antarctic_cells = []
        for i in range(N):
            pacific_cells.append((i, 0))
            antarctic_cells.append((i, M - 1))
        for j in range(M):
            pacific_cells.append((0, j))
            antarctic_cells.append((N - 1, j))
        propagate(pacific_cells)
        propagate(antarctic_cells)
        result = []
        for i in range(N):
            for j in range(M):
                if reached[i][j] == 2:
                    result.append([i, j])
        return result

    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        import heapq
        import math
        N = len(passingFees)
        graph = [[] for _ in range(N)]
        for e in edges:
            graph[e[0]].append((e[1], e[2]))
            graph[e[1]].append((e[0], e[2]))
        minTime = [math.inf] * N
        # cost, city, time
        heap = [(passingFees[0], 0, 0)]

        while heap:
            cost, city, time = heapq.heappop(heap)
            if city == N - 1: return cost
            if minTime[city] <= time: continue
            minTime[city] = time

            for edge in graph[city]:
                nei = edge[0]
                new_time = time + edge[1]
                new_cost = cost + passingFees[nei]
                if new_time > maxTime: continue
                heapq.heappush(heap, (new_cost, nei, new_time))
        return -1

    def timeTaken(self, arrival: List[int], state: List[int]) -> List[int]:
        from collections import deque
        persons = []
        for i in range(len(arrival)):
            persons.append((arrival[i], state[i], i))
        persons.sort()
        result = []
        index = 0
        process_time = 0
        exit_queue = deque()
        enter_queue = deque()
        previous_state = None

        def push_person(index):
            if persons[index][1] == 0:
                enter_queue.append(persons[index][2])
            else:
                exit_queue.append(persons[index][2])

        def process_once():
            if not enter_queue and not exit_queue: return None
            if not enter_queue:
                selected_queue = exit_queue
                next_state = 1
            elif not exit_queue:
                selected_queue = enter_queue
                next_state = 0
            elif previous_state is not None and previous_state == 0:
                selected_queue = enter_queue
                next_state = 0
            elif previous_state is not None and previous_state == 1:
                selected_queue = exit_queue
                next_state = 1
            else:
                selected_queue = exit_queue
                next_state = 1
            selected_index = selected_queue.popleft()
            result.append((selected_index, process_time))
            return next_state

        while True:
            while index < len(persons):
                if persons[index][0] <= process_time:
                    push_person(index)
                    index += 1
                else: break
            previous_state = process_once()
            process_time += 1
            if index == len(persons) and not enter_queue and not exit_queue: break

        final_result = [0] * len(arrival)
        for i, time in result:
            final_result[i] = time
        return final_result

    def wordPattern(self, pattern: str, s: str) -> bool:
        existing = dict()
        reverse_index = dict()
        words = s.split(' ')
        if len(pattern) != len(words): return False
        for i, p in enumerate(pattern):
            if p in existing:
                if existing[p] != words[i]: return False
                if reverse_index[words[i]] != p: return False
            else:
                if words[i] in reverse_index: return False
                existing[p] = words[i]
                reverse_index[words[i]] = p
        return True

    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        index = dict()
        reverse_index = dict()

        def check(i, j):
            if i == len(pattern):
                return j == len(s)
            if pattern[i] in index:
                w = index[pattern[i]]
                if s[j:j + len(w)] == w:
                    return check(i + 1, j + len(w))
                else: return False

            for k in range(j, len(s)):
                w = s[j:k + 1]
                if w in reverse_index: continue
                index[pattern[i]] = w
                reverse_index[w] = pattern[i]
                if check(i + 1, k + 1): return True
                del index[pattern[i]]
                del reverse_index[w]
            return False
        return check(0, 0)

    def kthCharacter(self, k: int) -> str:
        def gen(word):
            result = []
            for c in word:
                result.append(chr((ord(c) - ord('a') + 1) % 26 + ord('a')))
            return word + ''.join(result)
        word = 'a'
        while True:
            word = gen(word)
            if k - 1 < len(word):
                return word[k - 1]


    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        from collections import Counter
        counter = Counter()
        result = 0
        for w in words:
            for i in range(1, len(w) + 1):
                candidate = w[:i]
                if candidate != w[len(w) - i:]: continue
                if candidate not in counter: continue
                result += counter[candidate]
            counter[w] += 1
        return result

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        import math
        intervals = [(inter[0], inter[1]) for inter in intervals]
        intervals.sort()
        max_total = 0
        min_left = -math.inf

        for inter in intervals:
            if inter[0] >= min_left:
                min_left = inter[1]
                max_total += 1
            elif inter[1] < min_left:
                min_left = inter[1]
        return len(intervals) - max_total

    def maxDistance(self, colors: List[int]) -> int:
        first_min = 0
        second_min = None
        for i in range(1, len(colors)):
            if colors[i] != colors[first_min]:
                second_min = i
                break

        result = 0
        for i in range(1, len(colors)):
            if colors[i] == colors[first_min]:
                result = max(result, i - second_min)
            else:
                result = max(result, i - first_min)
        return result

    def maximumLength(self, s: str) -> int:
        def check(times):
            for i in range(26):
                c = chr(ord('a') + i)
                target = c * times
                index1 = s.find(target)
                if index1 == -1: continue
                tmp = s[index1 + 1:]
                index2 = tmp.find(target)
                if index2 == -1: continue
                index3 = tmp[index2 + 1:].find(target)
                if index3 != -1: return True
            return False

        left, right = 1, len(s) - 1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        if right == 0: return -1
        else: return right

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        import math
        if not root: return []
        rows = []
        stack = [(root, 0)]
        while stack:
            node, level = stack.pop()
            while len(rows) <= level:
                rows.append(-math.inf)
            rows[level] = max(rows[level], node.val)
            if node.left is not None:
                stack.append((node.left, level + 1))
            if node.right is not None:
                stack.append((node.right, level + 1))
        return rows

    def findMaxLength(self, nums: List[int]) -> int:
        max_length = 0
        s = 0
        seen_s = dict()
        for i, n in enumerate(nums):
            if n == 1:
                s += 1
            else:
                s -= 1
            if s == 0:
                max_length = max(max_length, i + 1)
            else:
                if s in seen_s:
                    max_length = max(max_length, i - seen_s[s])
                else:
                    seen_s[s] = i
        return max_length

    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if mid - 1 >= 0 and nums[mid] == nums[mid - 1]:
                if (mid + 1) % 2 == 0:
                    left = mid + 1
                else:
                    right = mid - 1
            elif mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                if (mid + 2) % 2 == 0:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                return nums[mid]
        return -1

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        next_greater = []
        result2 = [-1] * len(nums2)
        for i in range(len(nums2) - 1, -1, -1):
            while next_greater and next_greater[-1] <= nums2[i]:
                next_greater.pop()
            if not next_greater:
                result2[i] = -1
            else:
                result2[i] = next_greater[-1]
            next_greater.append(nums2[i])
        num2idx = {n: i for i, n in enumerate(nums2)}
        result = []
        for n in nums1:
            result.append(result2[num2idx[n]])
        return result

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        extended = nums + nums
        stack = []
        greater = [-1] * len(extended)
        for i in range(len(extended) - 1, -1, -1):
            while stack and stack[-1] <= extended[i]:
                stack.pop()
            if stack:
                greater[i] = stack[-1]
            stack.append(extended[i])
        return greater[:len(nums)]

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        max_depth = 0
        left_value = root.val

        stack = [(0, root)]
        while stack:
            depth, node = stack.pop()
            if depth > max_depth:
                max_depth = depth
                left_value = node.val
            if node.right is not None:
                stack.append((depth + 1, node.right))
            if node.left is not None:
                stack.append((depth + 1, node.left))
        return left_value

    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        from collections import Counter
        counter = Counter()

        def dfs(node):
            v = node.val
            if node.left is not None:
                v += dfs(node.left)
            if node.right is not None:
                v += dfs(node.right)
            counter[v] += 1
            return v
        dfs(root)
        max_val = max(counter.values())
        result = [k for k, v in counter.items() if v == max_val]
        return result

    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 1: return [[1]]
        rows = [[1], [1, 1]]
        if len(rows) == numRows: return rows
        while len(rows) < numRows:
            length = len(rows) + 1
            new_row = [1]
            while len(new_row) < length - 1:
                new_row.append(rows[-1][len(new_row) - 1] + rows[-1][len(new_row)])
            new_row.append(1)
            rows.append(new_row)
        return rows

    def getRow(self, rowIndex: int) -> List[int]:
        rowIndex += 1
        if rowIndex == 1: return [1]
        prev_row = [1, 1]
        index = 2
        if index == rowIndex: return prev_row
        while index < rowIndex:
            length = index + 1
            new_row = [1]
            while len(new_row) < length - 1:
                new_row.append(prev_row[len(new_row) - 1] + prev_row[len(new_row)])
            new_row.append(1)
            prev_row = new_row
            index += 1
        return prev_row

    def isPalindrome(self, s: str) -> bool:
        letters = set([chr(ord('a') + i) for i in range(26)]
                      + [str(i) for i in range(10)])
        a = []
        for c in s:
            c = c.lower()
            if c in letters:
                a.append(c)
        left, right = 0, len(a) - 1
        while left < right:
            if a[left] != a[right]: return False
            left += 1
            right -= 1
        return True

    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        map = dict()
        reverse_map = dict()
        for c_s, c_t in zip(s, t):
            if c_s in map:
                if map[c_s] != c_t: return False
            elif c_t in reverse_map:
                return False
            else:
                map[c_s] = c_t
                reverse_map[c_t] = c_s
        return True


    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode()
        node = head
        while node:
            current = node
            node = node.next
            current.next = dummy_head.next
            dummy_head.next = current
        return dummy_head.next

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(None)
        node = head
        tail = dummy_head
        while node:
            if tail.val != node.val:
                tail.next = node
                tail = node
                node = node.next
                tail.next = None
            else:
                node = node.next
        return dummy_head.next







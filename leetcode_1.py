import typing

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





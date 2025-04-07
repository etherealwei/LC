import leetcode_4
from leetcode_0_types import *

def build_interval(array):
    result = []
    for a in array:
        result.append(Interval(a[0], a[1]))
    return result


if __name__ == "__main__":
    solution = leetcode_4.Solution()
    print(solution.find_words(["abc", "a", "b", "c"], "abc"))

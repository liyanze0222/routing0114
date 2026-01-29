'''
977.有序数组的平方 的 Docstring
给你一个按 非递减顺序 排列的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排列。

递增的一次函数变为开口向上的抛物线函数，可以使用双指针从两侧遍历，比较平方值大小，填充结果数组。注意要从后往前填充结果数组，因为只能确定最大值在两侧。
'''

class Solution:
    def sortedSquares(self,nums:List[int]) -> List[int]:
        l,r,i = 0,len(nums) - 1,len(nums) - 1
        res = [0] * len(nums)
        while l <= r:
            if nums[l]*nums[l] < nums[r]*nums[r]:
                res[i] = nums[r]*nums[r]
                r = r - 1 #最大值在右侧，右指针左移
            else:
                res[i] = nums[l]*nums[l]
                l = l + 1 #最大值在左侧，左指针右移
            i = i - 1 #结果数组从后向前填充，因为前提是只知道最大值在两侧
        return res
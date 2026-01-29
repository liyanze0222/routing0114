'''
209.长度最小的子数组 的 Docstring
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

关键词是连续，因此要用滑动窗口；滑动窗口先延长，后收缩，延长靠右指针，满足条件就要收缩，靠左指针。注意：
1、更新最小长度的时候要在内层循环里
2、在非遍历完所有情况前，不能确定最终结果，因此最小长度初始值设为无穷大，且min_len取相对小值，最后判断是否更新过（这也是取无穷大的原因，可以判断是否更新）
'''

class Solution:
    def minSubArrayLen(self,s:int,nums:List[int]) -> int:
        left,right = 0,0
        min_len = float('inf')
        curr_sum = 0

        while right < len(nums):
            curr_sum = curr_sum + nums[right]

            while curr_sum >= s:
                min_len = min(min_len,right - left + 1)
                curr_sum = curr_sum - nums[left]
                left = left + 1

            right = right + 1

        return min_len if min_len != float('inf') else 0
'''
704.二分查找 的 Docstring
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。


'''

class Solution:
    def search(self,nums:List[int],target:int) -> int: #没有__init__，因为不需要存储属性
        left,right = 0,len(nums) - 1

        while left <= right:#不确定次数，用while循环
            middle = left + (right - left ) // 2 #取左中位数，避免溢出

            if nums[middle] > target:
                right = middle - 1 #target在左侧区间，不包含middle
            elif nums[middle] < target:
                left = middle + 1 #target在右侧区间，不包含middle
            else:
                return middle #找到了，返回索引值
        return -1 #没找到，返回-1
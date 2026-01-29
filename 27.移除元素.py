'''
27.移除元素 的 Docstring
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。你不需要考虑数组中超出新长度后面的元素。

标准的双指针例题，使用快慢指针实现。快指针用来扫描数组，慢指针有选择的重写元素，表面上像是在“移除”元素，实际上是覆盖掉需要移除的元素，从而达到“移除”的效果。
'''

class Solution: #双指针法
    def removeElement(self,nums:List[int],val:int)->int:
        fast = 0
        slow = 0
        size = len(nums)
        while fast < size:
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1 #没val，慢指针++
            fast += 1 #有没有val，快指针都++
        return slow
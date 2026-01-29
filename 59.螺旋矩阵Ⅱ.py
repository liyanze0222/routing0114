class Solution:
    def generateMatrix(self,n:int) -> List[List[int]]:
        nums = [[0] * n for _ in range(n)]
        '''
        这是一个习惯写法。当我们循环时不需要用到具体的索引值（比如 0, 1, 2...），
        只是为了单纯地重复动作时，就用 _ 来占位，表示“我不在乎这是第几次，反正你给我重复n次就行了”。
        ''' 
        startx,starty = 0,0
        loop,mid = n//2,n//2
        count = 1

        for offset in range(1,loop + 1): 
        #offset代表每一圈的缩进量 （大循环管圈数，小循环填数），loop+1则是为了让range包含最后一圈
            for i in range(starty,n - offset):
             #每一行/列只填n-1个数，最后一个数留给下一部分填；当然第二圈就是n-2个数，以此类推,便是n-offset
                nums[startx][i] = count
                count = count + 1
            for i in range(startx,n - offset):
                nums[i][n - offset] = count
                count = count + 1  
            for i in range(n - offset,starty,-1):
                nums[n - offset][i] = count
                count = count + 1   
            for i in range(n - offset,startx,-1):
                nums[i][starty] = count
                count = count + 1
            startx = startx + 1 
            starty = starty + 1
            #每一圈结束要把起始点向里斜向缩进一格
        if n % 2 == 1:
        # 如果n是奇数，则最中间的数没有被填过，需要单独处理
            nums[mid][mid] = count
        return nums

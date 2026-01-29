'''
44.开发商购买土地 的 Docstring
题目描述
某开发商准备购买一块土地用于开发建设。该土地被划分为 n 行 m 列的网格，每个网格单元有一个对应的价格。开发商希望通过在某一行或某一列上划分土地，将土地分成两部分，使得两部分土地的总价格之差的绝对值最小化。请你帮助开发商计算出这个最小的绝对差值。
输入描述
第一行包含两个整数 n 和 m，表示土地的行数和列数。
接下来的 n 行每行包含 m 个整数，表示每个网格单元的价格。
输出描述
输出一个整数，表示通过在某一行或某一列上划分土地所能得到的最小绝对差值。
示例输入
3 3
1 2 3
4 5 6
7 8 9
示例输出
1
'''

import sys
#为了读数据更快一些，使用 sys.stdin.read 读取所有输入，然后分割处理
def main():
    data = sys.stdin.read().split()

    idx = 0
    #即index做指针，不断向后移动读取数据
    n = int(data[idx])
    #先读行、列
    idx += 1
    m = int(data[idx])
    idx += 1
    
    sum = 0
    vec = []
    #准备读入矩阵，所以先创建一个空矩阵，这里的意思是先创建vec，然后一会一行一行读，一行一行填
    for i in range(n):
    #大循环，要读n行
        row = []
        #每行也得先创建一个空行
        for j in range(m):
        #小循环，每行m个数字
            val = int(data[idx])
            idx += 1
            row.append(val)
            sum += val
            #读、写、累加
        vec.append(row)
        #把读好的行放到矩阵里

    horizontal = [0] * n
    #开始做前缀和，由题意可知，各个横行要做，各个竖列要做
    for i in range(n):
    #有n行，所以这里range(n)
        for j in range(m):
            horizontal[i] += vec[i][j]

    vertical = [0] * m
    #同上
    for j in range(m):
        for i in range(n):
            vertical[j] += vec[i][j]

    result = float('inf')
    #两部分前缀和做完了，接下来开始解题，题目要求输出最小绝对差值，而易知需要遍历切割情况，所以初始化一个无穷大
    
    horizontal_sum = 0
    #这是切割点以上的和，这里的思路是总和sum减去2倍的切割点以上的和，就是两部分的差值
    for i in range(n):
        horizontal_sum += horizontal[i]
        result = min(result, abs(sum - 2 * horizontal_sum))

    vertical_sum = 0
    for j in range(m):
        vertical_sum += vertical[j]
        result = min(result, abs(sum - 2 * vertical_sum))

    print(result)
if __name__ == "__main__":
    main()
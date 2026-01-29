import sys

def main():
    # 首先读进来所有的数据（字符串形式）
    data = sys.stdin.read().split()
    #把数据按空格/换行符分割成一个列表
    index = 0
    n = int(data[index])
    #获取第一个数据，数组长度
    index = index + 1
    vec = []
    #准备获取数组元素，构建一个n次的循环（0~n-1）
    for i in range(n):
        vec.append(int(data[index+i]))
        #注意int把字符串转成整数
    index = index + n

    p = [0] * n
    #在一维数组中，可以用这样的形式创建一个有n个0元素的数组
    presum = 0
    #前缀和初始化
    for i in range(n):
        presum = presum + vec[i]
        p[i] = presum
    
    results = []
    #因为不知道有多少个result需要算，所以先创建一个空列表
    while index < len(data):
        a = int(data[index])
        b = int(data[index+1])
        index = index + 2
        #读取每一对a,b，对每一对进行计算，原理类似数列的求和
        if a == 0:
            sum_value = p[b]
        else:
            sum_value = p[b] - p[a-1]

        results.append(str(sum_value))
        #转成字符串，一会输出

    for result in results:
    #此处result是临时变量，代表results列表中的每一个元素，一个一个打印出来
        print(result)

if __name__ == "__main__":
    main()
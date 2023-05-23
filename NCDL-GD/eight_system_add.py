def add(a, b, c):
    # m_list = list('01234567');
    # n_list = list('1');
    m_list = a;
    n_list = b;
    m_list.reverse() # 翻转，方便从个位开始加
    n_list.reverse()
    # 加数和被加数补齐，防止数组越界，短者高位补0
    if len(m_list) > len(n_list):
        result = [''] * (len(m_list) + 1)# 保存和，多一位是防止最高位也有进一的情况
        n_list = n_list + [0] * (len(m_list) - len(n_list))
    else:
        result = [''] * (len(n_list) + 1)
        m_list = m_list + [0] * (len(n_list) - len(m_list))

    flag = False
    for i in range(max(len(m_list), len(n_list))):
        if flag: # 如果上一位有进1，本位和需要加上上一位进的1
            plus = int(n_list[i]) + int(m_list[i]) + 1
        else:
            plus = int(n_list[i]) + int(m_list[i])

        if plus >= 8: # 本位大于8，本位存本位和-8，并向前进一
            result[i] = str(plus - 8)
            flag = True
        else:
            result[i] = str(plus)
            flag = False
    if flag: # 最高位最终向前进1，和也需要向前进1
        result[-1] = str(1)#python中数组下标为-1时 https://blog.csdn.net/jiayizhenzhenyijia/article/details/97623762
                           #result的下标从0到5，最后一个元素是result[5]，但最后一个元素也可以用result[-1]来表示
    result.reverse()
    int_result = map(int,result)
    print(''.join(result).strip())
    return c

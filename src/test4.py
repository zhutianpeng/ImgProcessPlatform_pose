from RedisQueue import RedisQueue

q3 = RedisQueue('test1')  #发送队列
while q3.empty() == False:
    tup1 = tuple(eval(q3.get()))
    print(tup1[0],tup1[1],tup1)


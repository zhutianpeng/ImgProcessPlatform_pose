from RedisQueue import RedisQueue
import dealimg

q = RedisQueue('test')

i=0
while q.empty()==False:
    base64_data = q.get()
    dealimg.image_decode(base64_data,i)
    print(i)
    i=i+1
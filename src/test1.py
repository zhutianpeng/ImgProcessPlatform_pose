from RedisQueue import RedisQueue

q_face = RedisQueue('face_forward_list')  #face发送队列
q_pose = RedisQueue('pose_forward_list')  #pose发送队列
q2 = RedisQueue('test_return_list')   #接收队列
DB_set = RedisQueue('return_set')        #缓存的set
q_face.delall()
q_face.delall()
q2.delall()
DB_set.delall()
import redis

class RedisQueue(object):
    """Simple Queue with Redis Backend"""
    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.key = '%s:%s' %(namespace, name)

    def delall(self):
        """del all this queue items."""
        self.__db.delete(self.key)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)

        if item:
            item = item[1]
        return item

    # set的增加操作
    def setSet(self, name, key, value):
        self.__db.hset(name,key,value)

    # set的获取操作
    def getSet(self, name,key):
        item = self.__db.hget(name,key)
        return item

    def sSize(self, name):
        # item = self.__db.scard(name)
        item = self.__db.hlen(name)
        return item


    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)
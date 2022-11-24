import sys
from typing import Dict
import numpy as np
import collections
from sklearn.neighbors import NearestNeighbors
available_engines = {"sklearn"}
try:
    import hnswlib
    available_engines.add("hnswlib")
except ModuleNotFoundError:
    print ("hnswlib not found")
    HNSWMock = collections.namedtuple("HNSWMock", ("Index", "max_elements"))
    class MockHnsw:
        def __init__(self, *args, **kwargs) -> None:
            pass
    hnswlib = HNSWMock(MockHnsw(),0)
try:
    import faiss
    available_engines.add("faiss")
except Exception:
    print ("faiss not found")
    faiss = None
try:
    from redis import Redis
    from redis.commands.search.field import VectorField
    from redis.commands.search.field import TextField
    from redis.commands.search.field import TagField
    from redis.commands.search.query import Query
    from redis.commands.search.result import Result
    available_engines.add("redis")
except ModuleNotFoundError:
    print ("redis not found")

def parse_server_name(sname):
    if sname in ["hnswlib", "hnsw"]:
        if "hnswlib" in available_engines:
            return LazyHnsw
        return SciKitNearestNeighbors
    elif sname in ["faiss", "flatfaiss"]:
        if "faiss" in available_engines:
            return FaissIndexFactory
        return SciKitNearestNeighbors
    elif sname in ["redis"]:
        if "redis" in available_engines:
            return RedisIndex
        return SciKitNearestNeighbors
    else:
        return SciKitNearestNeighbors


class FaissIndexFactory:
    def __init__(self, space:str, dim:int, index_factory:str, **kwargs):
        if index_factory == '':
            index_factory = 'Flat'
        if space in ['ip', 'cosine']:
            self.index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
            if space == 'cosine':
                # TODO: Support cosine
                sys.stderr.write("cosine is not supported yet, falling back to dot\n")
        elif space == 'l2':
            self.index = faiss.index_factory(dim, index_factory, faiss.METRIC_L2)
        else:
            raise TypeError(str(space) + " is not supported")
        self.index = faiss.IndexIDMap2(self.index)

    def __len__(self):
        return self.get_current_count()

    def  __itemgetter__(self, item):
        return self.index.reconstruct(int(item))
    
    def add_items(self, data, ids):
        data = np.array(data).astype(np.float32)
        self.index.add_with_ids(data, np.array(ids, dtype='int64'))

    def get_items(self, ids):
        # recmap = {k:i for i,k in enumerate(faiss.vector_to_array(self.index.id_map))}
        # return np.vstack([self.index.reconstruct(recmap[v]) for v in ids])
        return np.vstack([self.index.reconstruct(int(v)) for v in ids])

    def get_max_elements(self):
        return -1

    def get_current_count(self):
        return self.index.ntotal

    def search(self, data, k=1):
        return self.index.search(np.array(data).astype(np.float32),k)

    def save_index(self, fname):
        return faiss.write_index(self.index, fname)

    def load_index(self, fname):
        self.index = faiss.read_index(fname)

class LazyHnsw(hnswlib.Index):
    def __init__(self, space:str, dim:int, max_elements=1024, ef_construction=200, M=16,**kwargs):
        super().__init__(space, dim)
        self.init_max_elements = max_elements
        self.init_ef_construction = ef_construction
        self.init_M = M

    def __len__(self):
        return self.get_current_count()

    def  __itemgetter__(self, item):
        return super().get_items([item])[0]

    def init(self, max_elements=0):
        if max_elements == 0:
            max_elements = self.init_max_elements
        super().init_index(max_elements, self.init_M, self.init_ef_construction)

    def add_items(self, data, ids=None, num_threads=-1):
        if self.max_elements == 0:
            self.init()
        if self.max_elements<len(data)+self.element_count:
            super().resize_index(max([len(data)+self.element_count,self.max_elements]))
        return super().add_items(data, ids, num_threads)

    def add(self, data):
        return self.add_items(data)

    def get_items(self, ids=None):
        if self.max_elements == 0:
            return []
        return super().get_items(ids)

    def knn_query(self, data, k=1, num_threads=-1):
        if self.max_elements == 0:
            return [], []
        return super().knn_query(data, k, num_threads)

    def search(self, data, k=1):
        I,D = self.knn_query(data, k)
        return (D,I)

    def resize_index(self, size):
        if self.max_elements == 0:
            return self.init(size)
        else:
            return super().resize_index(size)

    def set_ef(self, ef):
        if self.max_elements == 0:
            self.init_ef_construction = ef
            return
        super().set_ef(ef)

    def get_max_elements(self):
        return self.max_elements

    def get_current_count(self):
        return self.element_count


class SciKitNearestNeighbors:
    def __init__(self, space:str, dim:int, **kwargs):
        if space=="ip":
            self.space = "cosine"
            sys.stderr.write("Warning: ip is not supported by sklearn, falling back to cosine")
        else:
            self.space = space
        self.dim = dim
        self.items = []
        self.ids = []
        self.fitted = False
        self.index = NearestNeighbors(metric=self.space,n_jobs=-1,n_neighbors=10)

    def __len__(self):
        return len(self.items)

    def  __itemgetter__(self, item):
        return self.items[self.ids.index(item)]

    def init(self, **kwargs):
        self.index.fit(self.items)
        self.fitted = True
        
    def add_items(self, data, ids=None, num_threads=-1):
        self.items.extend(data)
        if ids is None:
            ids = list(range(len(self.items),len(self.items)+len(data)))
        self.ids.extend(ids)
        self.fitted = False

    def get_items(self, ids=None):
        return [self.items[self.ids.index(i)] for i in ids]

    def search(self, data, k=1):
        if not self.fitted:
            self.index.fit(self.items)
            self.fitted = True
        scores, idx = self.index.kneighbors(data ,k, return_distance=True)
        names = [[self.ids[i] for i in ids] for ids in idx]
        return scores, names

    def get_max_elements(self):
        return -1

    def get_current_count(self):
        return len(self.items)


class RedisIndex:
    def __init__(self, space:str, dim:int, redis_credentials=None,max_elements=1024, ef_construction=200, M=16, overwrite=True,**kwargs):
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        if kwargs.get("index_name") is None:
            self.index_name = "idx"
        else:
            self.index_name = kwargs.get("index_name")
        if redis_credentials is None:
            raise Exception("Redis credentials must be provided")
        self.redis = Redis(**redis_credentials)
        self.pipe = None
        if overwrite:
            try:
                self.redis.ft(self.index_name).info()
                index_exists = True
            except:
                index_exists = False
            if index_exists:
                self.redis.ft(self.index_name).dropindex(delete_documents=True)
        self.init_hnsw()
        # applicable only for user events
        self.user_keys=[]
    
    def __enter__(self):
        self.pipe = self.redis.pipeline()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe.execute()
        self.pipe = None

    def __len__(self):
        return self.get_current_count()

    def  __itemgetter__(self, item):
        return super().get_items([item])[0]

    def user_keys(self):
        """Get all user keys"""
        return [s.decode()[5:] for s in self.redis.keys("user:*")]

    def item_keys(self):
        """Get all item keys"""
        return [s.decode()[5:] for s in self.redis.keys("item:*")]

    def vector_keys(self):
        """Get all vector keys"""
        return [s.decode()[4:] for s in self.redis.keys("vec:*")]

    def search(self, data, k=1,partition=None):
        """Search the nearest neighbors of the given vectors, and a given partition."""
        query_vector = np.array(data).astype(np.float32).tobytes()
        #prepare the query
        p = "(@partition:{"+partition+"})" if partition is not None else "*"
        q = Query(f'{p}=>[KNN {k} @embedding $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).return_fields('vector_score','item_id').dialect(2)
        params_dict = {"vec_param": query_vector}
        results = self.redis.ft(self.index_name).search(q, query_params = params_dict)
        scores, ids = [], []
        for item in results.docs:
            scores.append(item.vector_score)
            ids.append(item.item_id)
        return scores, ids

    def add_items(self, data, ids=None, partition=None):
        """Add items and ids to the index, if a partition is not defined it defaults to NONE"""
        self.pipe = self.redis.pipeline(transaction=False)
        if partition is None:
            partition="NONE"
        for datum, id in zip(data, ids):
            key='item:'+ str(id)
            emb = np.array(datum).astype(np.float32).tobytes()
            self.pipe.hset(key,mapping={"embedding": emb, "item_id": str(id), "partition":partition})
                
        self.pipe.execute()
        self.pipe = None

    def get_items(self, ids=None):
        """Get items by id"""
        ret = []
        for id in ids:
            ret.append(np.frombuffer(self.redis.hget("item:"+str(id), "embedding"), dtype=np.float32))
        return np.vstack(ret)

    def add_user_event(self, user_id: str, data: Dict[str, str],ttl: int = 60*60*24):
        """
        Adds a user event to the index. The event is stored in a hash with the key user:{user_id} and the fields
        fields are defined by the `user_keys` property
        """
        if not any(self.user_keys):
            raise Exception("User keys must be set before adding user events")
        vals = []
        for key in self.user_keys:
            v = data.get(key,"")
            # ad hoc int trimming
            try:
                if v==int(v):
                    v=int(v)
            except:
                pass
            vals.append(v)
        val = '|'.join(map(str, vals))
        if self.pipe:
            self.pipe.rpush("user:"+str(user_id), val)
            if ttl:
                self.pipe.expire("user:"+str(user_id), ttl)
        else:
            self.redis.rpush("user:"+str(user_id), val)
            if ttl:
                self.redis.expire("user:"+str(user_id), ttl)
        return self
    def del_user(self, user_id):
        """Delete a user key from Redis"""
        if self.pipe:
            self.pipe.delete("user:"+str(user_id))
        else:
            self.redis.delete("user:"+str(user_id))
    
    def get_user_events(self, user_id: str):
        """Gets a list of user events by key"""
        if not any(self.user_keys):
            raise Exception("User keys must be set before getting user events")
        ret = self.redis.lrange("user:"+str(user_id), 0, -1)
        return [dict(zip(self.user_keys,x.decode().split('|'))) for x in ret]
    
    def set_vector(self, key, arr, prefix="vec:"):
        """Sets a numpy array as a vector in redis"""
        emb = np.array(arr).astype(np.float32).tobytes()
        self.redis.set(prefix+str(key), emb)
        return self
    
    def get_vector(self, key, prefix="vec:"):
        """Gets a numpy array from redis"""
        return np.frombuffer(self.redis.get(prefix+str(key)), dtype=np.float32)


    def init_hnsw(self, **kwargs):
        self.redis.ft(self.index_name).create_index([
        VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": self.space, "INITIAL_CAP": self.max_elements, "M": self.M, "EF_CONSTRUCTION": self.ef_construction}),
        TextField("item_id"),
        TagField("partition")
        ])  

    def get_current_count(self):
        """Get number of items in index"""
        return int(self.redis.ft(self.index_name).info()["num_docs"])

    def get_max_elements(self):
        """Get max elements in index"""
        return self.max_elements
    
    def info(self):
        """Get Redis info as dict"""
        return self.redis.ft(self.index_name).info()

if __name__=="__main__":
    # docker run -p 6379:6379 redislabs/redisearch:2.4.5
    sim = RedisIndex(space='cosine',dim=32,redis_credentials={"host":"127.0.0.1", "port": 6379}, overwrite=True)
    data=np.random.random((100,32))
    aids=["a"+str(1+i) for i in range(100)]
    bids=["b"+str(101+i) for i in range(100)]
    sim.add_items(data,aids,partition="a")
    sim.add_items(data,bids,partition="b")
    # print(sim.search(data[0],k=10,partition=None))
    # print(sim.get_items(aids[:10]))
    print (sim.item_keys())
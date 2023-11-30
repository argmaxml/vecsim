import sys, json
from typing import Dict
import numpy as np
import collections
import itertools
from sklearn.neighbors import NearestNeighbors
available_engines = {"sklearn"}
try:
    import faiss
    available_engines.add("faiss")
except Exception:
    faiss = None
try:
    import pinecone
    available_engines.add("pinecone")
except Exception:
    pinecone = None
try:
    from redis import Redis
    from redis.commands.search.field import VectorField
    from redis.commands.search.field import TextField
    from redis.commands.search.field import TagField
    from redis.commands.search.query import Query
    from redis.commands.search.result import Result
    available_engines.add("redis")
except ModuleNotFoundError:
    Redis = None

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    available_engines.add("elastic")
except ModuleNotFoundError:
    Elasticsearch = None


class BaseIndex:
    def __init__(self, metric:str, dim:int):
        self.dim = dim
        self.metric = metric
        self.fitted = False
        # Per partition mapping
        self.items = collections.defaultdict(list)
        self.ids = collections.defaultdict(list)
        self.indices = collections.defaultdict(lambda: None)
        self.cls = None
        self.cls_args = {}
    def get_partitions(self):
        return list(self.items.keys())
    def init(self, **kwargs):
        kwargs = {**self.cls_args, **kwargs}
        if hasattr(self.cls, "fit"):
            self.indices = {p:self.cls(**kwargs) for p in self.indices.keys()}
        for p,index in self.indices.items():
            if hasattr(index, "fit"):
                index.fit(self.items[p])
            elif hasattr(index, "init"):
                index.init()
        self.fitted = True
    def add_items(self, data, ids=None, partition=None):
        # Support for adding items to multiple partitions
        if type(partition)==np.str_:
            partition = str(partition)
        if hasattr(partition, "__iter__") and type(partition)!=str:
            data = np.array(data)
            ids = np.array(ids)
            ps = set(partition)
            partition = np.array(partition)
            for p in ps:
                self.add_items(data[partition==p], ids[partition==p], p)
            return
        # Support for adding items to a single partition
        if ids is None:
            ids = list(range(len(self.items[partition]),len(self.items[partition])+len(data)))
        elif type(ids)==np.ndarray:
            ids = ids.tolist()
        if hasattr(self.cls, 'add_items'):
            self.ids[partition].extend(ids)
            self.indices[partition].add_items(data)
        else:
            self.items[partition].extend(data)
            self.indices[partition]=None
            n_before = len(set(self.ids[partition]))
            self.ids[partition].extend(ids)
            n_after = len(set(self.ids[partition]))
            if n_after != n_before+len(ids):
                raise ValueError("ids must be unique")
        self.fitted = False

    def get_items(self, ids=None,partition=None):
        if type(ids)==str:
            raise SyntaxError("ids must be a list")
        if hasattr(self.indices[partition], "get_items"):
            return self.indices[partition].get_items(ids)
        return [self.items[partition][self.ids[partition].index(i)] for i in ids]

    def search(self, data, k=1,partition=None):
        if not self.fitted:
            if hasattr(self.indices[partition], "fit"):
                self.indices[partition].fit(self.items[partition])
            elif hasattr(self.indices[partition], "init"):
                self.indices[partition].init()
        if data.ndim==1:
            data = data.reshape(1,-1)
        if partition:
            scores_p, idx_p = self.indices[partition].search(data, k)
            scores, ids = scores_p[0], idx_p[0]
            names = [self.ids[partition][i] for i in ids]
        else:
            scores, names = [], []
            for p in self.indices:
                try:
                    scores_p, idx_p = self.indices[p].search(data, k)
                except ValueError:
                    continue
                if len(scores_p)==0:
                    continue
                scores_p, idx_p = scores_p[0], idx_p[0]
                scores.extend(scores_p)
                names.extend([self.ids[p][i] for i in idx_p])
            if len(names)>k:
                sorter = np.argsort(scores)
                scores = np.array(scores)[sorter].tolist()[:k]
                names = np.array(names)[sorter].tolist()[:k]
        return scores, names

    def search_and_aggregate(self, data, k=1,partition=None,aggfunc='len',delim="|",reverse=False):
        if type(aggfunc)==str:
                if aggfunc == "len":
                    aggfunc = len
                    reverse = True
                if aggfunc in dir(np):
                    aggfunc = getattr(np, aggfunc)
        scores, ids = self.search(data, k,partition)
        if len(scores)==0:
            return [], []
        scores_lst = collections.defaultdict(list)
        for score, id in zip(scores, ids):
            grp_id = id.split(delim,1)[0]
            scores_lst[grp_id].append(score)
        ret = {}
        for grp_id in scores_lst:
            ret[grp_id] = aggfunc(scores_lst[grp_id])
        ret = sorted(ret.items(), key=lambda x: x[1], reverse=reverse)
        ids, scores = zip(*ret)
        return scores, ids
    def __repr__(self):
        partitions = ",".join(map(str, self.items.keys()))
        return f"{self.cls.__name__}(dim={self.dim}, metric={self.metric}, partitions={partitions})"
    def __str__(self):
        return self.__repr__()
    def __len__(self):
        length = 0
        for p in self.items:
            length += len(self.items[p])
        return length
    def  __itemgetter__(self, item):
        for p in self.items:
            if item in self.ids[p]:
                return self.items[p][self.ids[p].index(item)]
        return None

class SciKitIndexUnpartitioned(NearestNeighbors):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def search(self, data, k=1):
        if k>self.n_neighbors:
            sys.stderr.write("Warning: k is larger than n_neighbors, falling back to n_neighbors\n")
            k = self.n_neighbors
        return super().kneighbors(data, k, return_distance=True)

class SciKitIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, k=10, **kwargs):
        BaseIndex.__init__(self, metric, dim)
        if metric=="ip":
            self.metric = "cosine"
            sys.stderr.write("Warning: ip is not supported by sklearn, falling back to cosine")
        self.cls = SciKitIndexUnpartitioned
        self.cls_args={"n_neighbors":k, "metric":self.metric, "n_jobs":-1}
        self.indices = collections.defaultdict(lambda:self.cls(**self.cls_args))


class RedisIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, redis_credentials=None,max_elements=1024, ef_construction=200, M=16, overwrite=True,**kwargs):
        BaseIndex.__init__(self, metric, dim)
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.partitions = set()
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
        try:
            self.redis.ft(self.index_name).info()
        except:
            raise Exception("Index not found, please initialize the index by setting overwrite to True")
        # applicable only for user events
        self.user_keys=[]
    
    def init(self):
        pass

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

    def get_user_keys(self):
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
        if partition is not None and partition not in self.partitions:
            raise Exception(f"Partition {partition} does not exist")
        query_vector = np.array(data).astype(np.float32).tobytes()
        #prepare the query
        p = "(@partition:{"+partition+"})" if partition is not None else "*"
        q = Query(f'{p}=>[KNN {k} @embedding $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).return_fields('vector_score','item_id').dialect(2)
        params_dict = {"vec_param": query_vector}
        results = self.redis.ft(self.index_name).search(q, query_params = params_dict)
        scores, ids = [], []
        for item in results.docs:
            scores.append(float(item.vector_score))
            ids.append(item.item_id)
        return scores, ids

    def add_items(self, data, ids=None, partition=None):
        """Add items and ids to the index, if a partition is not defined it defaults to NONE"""
        # Support for adding items to multiple partitions
        if type(partition)==np.str_:
            partition = str(partition)
        if hasattr(partition, "__iter__") and type(partition)!=str:
            data = np.array(data)
            ids = np.array(ids)
            ps = set(partition)
            partition = np.array(partition)
            for p in ps:
                self.add_items(data[partition==p], ids[partition==p], p)
            return
        # Support for adding items to a single partition
        self.pipe = self.redis.pipeline(transaction=False)
        if partition is None:
            partition="NONE"
        self.partitions.add(partition)
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
    
    def set_vector(self, key, arr, prefix="vec:", ttl:int=60*60*24):
        """Sets a numpy array as a vector in redis"""
        emb = np.array(arr).astype(np.float32).tobytes()
        self.redis.set(prefix+str(key), emb)
        if ttl:
            self.redis.expire(prefix+str(key), ttl)
        return self
    
    def get_vector(self, key, prefix="vec:"):
        """Gets a numpy array from redis"""
        return np.frombuffer(self.redis.get(prefix+str(key)), dtype=np.float32)


    def init_hnsw(self, **kwargs):
        self.redis.ft(self.index_name).create_index([
        VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": self.metric, "INITIAL_CAP": self.max_elements, "M": self.M, "EF_CONSTRUCTION": self.ef_construction}),
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

class ElasticIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, elastic_credentials,index_name="vecsim",M=16,ef_construction=100,**kwargs):
        self.es = Elasticsearch(**elastic_credentials)
        self.index_name = index_name
        self.partitions = set()
        if metric=="ip":
            metric="dot_product"
        elif metric=="l2":
            metric="l2_norm"
        mappings =  {
            "properties": {
                "vec": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": metric,
                    "index_options": {
                        "type": "hnsw",
                        "m": M,
                        "ef_construction": ef_construction
                    }
                },
                "partition" : {
                    "type": "text",
                    "analyzer": "keyword"
                    },
                "id": {
                    "type": "text",
                    "analyzer": "keyword"
                },
                },
                # "text" : {
                #     "type" : "keyword"
                # }
        }
            
        self.es.indices.create(index=index_name, ignore=400, body={"mappings":mappings})
    
    def add_items(self, data, ids=None, partition=None):
        # Support for adding items to multiple partitions
        if type(partition)==np.str_:
            partition = str(partition)
        if hasattr(partition, "__iter__") and type(partition)!=str:
            data = np.array(data)
            ids = np.array(ids)
            ps = set(partition)
            partition = np.array(partition)
            for p in ps:
                self.add_items(data[partition==p], ids[partition==p], p)
            return
        # Support for adding items to a single partition
        self.partitions.add(partition)
        bulk(self.es, [{
                "_index": self.index_name,
                "_source": {"vec": [float(x) for x in datum], "id": str(id), "partition":partition},
            } for datum, id in zip(data, ids)])
    
    def search(self, data, k=1,partition=None, num_candidates=50):
        if partition is not None and partition not in self.partitions:
            raise Exception(f"Partition {partition} does not exist")
        if partition:
            query = {"query": {"bool": {"must": [{"match": {"partition": partition}}, {"knn": {"field":"vec","query_vector": [float(x) for x in data], "k": k, "num_candidates": num_candidates}}]}}}
        else:
            query = {"knn": {"field":"vec","query_vector": [float(x) for x in data], "k": k, "num_candidates": num_candidates}}
        res = self.es.search(index=self.index_name, body=query)
        ids = [x["_source"]["id"] for x in res["hits"]["hits"]]
        scores = [x["_score"] for x in res["hits"]["hits"]]
        return scores, ids

    def init(self, **kwargs):
        pass

    def get_items(self, ids=None):
        """Get items by id"""
        if ids is None:
            ids = []
        if not isinstance(ids, list):
            ids = [ids]
        res = self.es.mget(index=self.index_name, body={"ids": ids})
        return [np.array(x["_source"]["vec"]) for x in res["docs"] if x["found"]]

    def get_current_count(self):
        """Get number of items in index"""
        return self.es.count(index=self.index_name)["count"]
    
    def __len__(self):
        return self.get_current_count()

    def  __itemgetter__(self, item):
        return super().get_items([item])[0]



## Faiss

class FaissBase:
    def __init__(self, metric:str, dim:int):
        self.dim = dim
        self.metric = metric
        self.index = None
        self.data = None
        
    def add_items(self, data):
        data = np.array(data).astype(np.float32)
        if self.metric=='cosine':
            faiss.normalize_L2(data)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(data)
        # self.data = data
        self.index.add(data)

    def init(self):
        pass

    def get_current_count(self):
        return self.index.ntotal

    def search(self, data, k=1):
        data = np.array(data).astype(np.float32)
        if self.metric=='cosine':
            faiss.normalize_L2(data)
            # data/=np.linalg.norm(data,axis=1)
        return self.index.search(data,k)

    def save_index(self, fname):
        return faiss.write_index(self.index, fname)

    def load_index(self, fname):
        self.index = faiss.read_index(fname)
    def __len__(self):
        return self.get_current_count()
class FaissFlatIndexUnpartitioned(FaissBase):
    def __init__(self, metric:str, dim:int):
        self.dim = dim
        self.metric = metric
        if metric in ['ip', 'dot']:
            self.index = faiss.IndexFlatIP(self.dim)
        elif metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.dim)
        elif metric == 'l2':
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise TypeError(str(metric) + " is not supported")

class FaissIVFPQIndexUnpartitioned(FaissFlatIndexUnpartitioned):
    def __init__(self, metric:str, dim:int, nlist:int=256, m:int=8, nbits:int=8):
        super().__init__(metric, dim)
        self.index = faiss.IndexIVFPQ(self.index, self.dim, nlist, m, nbits)


class FaissIndexFactoryUnpartitioned(FaissBase):
    def __init__(self, metric:str, dim:int, index_factory:str, **kwargs):
        self.dim = dim
        self.metric = metric
        if index_factory == '':
            index_factory = 'Flat'
        if metric in ['ip', 'dot', 'cosine']:
            self.index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
        elif metric == 'l2':
            self.index = faiss.index_factory(dim, index_factory, faiss.METRIC_L2)
        else:
            raise TypeError(str(metric) + " is not supported")

    def  __itemgetter__(self, item):
        return self.index.reconstruct(int(item))

    def get_items(self, ids):
        # recmap = {k:i for i,k in enumerate(faiss.vector_to_array(self.index.id_map))}
        # return np.vstack([self.index.reconstruct(recmap[v]) for v in ids])
        return np.vstack([self.index.reconstruct(int(v)) for v in ids])



class FaissIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, index_factory:str, **kwargs):
        BaseIndex.__init__(self, metric, dim)
        self.cls = FaissIndexFactoryUnpartitioned
        self.cls_args={"metric":self.metric,"dim":self.dim,"index_factory":index_factory}
        self.indices = collections.defaultdict(lambda:self.cls(**self.cls_args))

class FaissFlatIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, **kwargs):
        BaseIndex.__init__(self, metric, dim)
        self.cls = FaissFlatIndexUnpartitioned
        self.cls_args={"metric":self.metric,"dim":self.dim}
        self.indices = collections.defaultdict(lambda:self.cls(**self.cls_args))

class FaissIVFPQIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, **kwargs):
        BaseIndex.__init__(self, metric, dim)
        self.cls = FaissIVFPQIndexUnpartitioned
        self.cls_args={"metric":self.metric,"dim":self.dim}
        self.indices = collections.defaultdict(lambda:self.cls(**self.cls_args))

## Pinecone
class PineconeIndex(BaseIndex):
    def __init__(self, metric:str, dim:int, pinecone_credentials, index_name="vecsim",**kwargs):
        self.index_name = index_name
        self.partitions = set()
        if metric == "cosine":
            metric = 'cosine'
        elif metric in ['ip', 'dot']:
            metric = 'dotproduct'
        elif metric == 'l2':
            metric = 'euclidean'
        else:
            raise TypeError(str(metric) + " is not supported")
        pinecone.init(**pinecone_credentials)
        try:
            pinecone.describe_index(index_name)
        except:
            pinecone.create_index(index_name, dimension=dim, metric=metric, **kwargs)
        self.index = pinecone.Index(index_name)

    def _get_data_chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def add_items(self, data, ids, partition=None):
        # Support for adding items to multiple partitions
        if type(partition)==np.str_:
            partition = str(partition)
        if hasattr(partition, "__iter__") and type(partition)!=str:
            data = np.array(data)
            ids = np.array(ids)
            ps = set(partition)
            partition = np.array(partition)
            for p in ps:
                self.add_items(data[partition==p], ids[partition==p], p)
            return
        self.partitions.add(partition)
        if partition is not None:
            data_generator = map(lambda i: (
                ids[i], 
                [float(item) for item in data[i]],
                {"partition": partition}
                ), range(len(ids)))
        else:
            data_generator = map(lambda i: (
                            str(ids[i]), 
                            [float(item) for item in data[i]]
                            ), range(len(ids)))
            
        for ids_vectors_chunk in self._get_data_chunks(data_generator, batch_size=100):
            self.index.upsert(vectors=list(ids_vectors_chunk))

    
    def search(self, data, k=1,partition=None):
        if partition is not None and partition not in self.partitions:
            raise Exception(f"Partition {partition} does not exist")
        query_args = {
            "vector": [float(i) for i in data],
            "top_k": k,
        }
        if partition:
            query_args["filter"] = {"partition": {"$eq": partition}}
        response = self.index.query(**query_args)
        matches = response['matches']
        ids = [x["id"] for x in matches]
        scores = [x["score"] for x in matches]
        return scores, ids
    
    def get_items(self, ids):
        """Get items by id"""
        if not isinstance(ids, list):
            ids = [ids]
        res = self.index.fetch(ids)
        return [np.array(res['vectors'][x]["values"]) for x in res['vectors'].keys()]

    def get_current_count(self):
        """Get number of items in index"""
        index_stats = self.index.describe_index_stats()
        return index_stats['total_vector_count']
    



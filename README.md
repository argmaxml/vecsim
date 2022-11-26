# VecSim - an unified interface for similarity servers
A standard, light-weight interface to all popular similarity servers.

## The problems we are trying to solve:
1. **Standard API** - Different vector similarity servers has different APIs - so switching is not trivial.
1. **Identifiers** - Some vector similarity servers support string IDs, some do not - we keep track of the mapping.
1. **Partitions** - In most cases, a pre-filtering is needed prior to querying, we abstract this concept away.

## Supported engines:
1. Scikit-learn, via [NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
1. [RediSearch](https://redis.io/docs/stack/search/reference/vectors/)
1. [Faiss](https://github.com/facebookresearch/faiss)


## QuickStart example
```python
import numpy as np
# Import a similarity server of your choice:
# SKlearn (best for small datasets or testing)
from vecsim import SciKitIndex
sim = SciKitIndex(metric='cosine', dim=32)
# OR Faiss:
# from vecsim import FaissIndex
# sim = FaissIndex(metric='ip', dim=32, index_factory='L2norm,Flat')
# OR Redis:
# from vecsim import RedisIndex
# sim = RedisIndex(metric='cosine', dim=32, redis_credentials={"host":"127.0.0.1", "port": 6379}, overwrite=True)
# Feed in some vectors for both users and items
user_ids = ["user_"+str(1+i) for i in range(100)]
user_data = np.random.random((100,32))
item_ids=["item_"+str(101+i) for i in range(100)]
item_data = np.random.random((100,32))
sim.add_items(user_data,user_ids,partition="users")
sim.add_items(item_data,item_ids,partition="items")
# Index the data
sim.init()
# Run nearest neighbor vector search
query = np.random.random(32)
dists, items = sim.search(query,k=10) # returns a list of users and items
dists, items = sim.search(query,k=10,partition="users") # returns a list of only users
```

For more examples, please read our [documentation](https://vecsim.readthedocs.io/)
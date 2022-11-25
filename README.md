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
from vecsim import SciKitIndex
sim = SciKitIndex(metric='cosine', dim=32)
# Feed in some vectors for both customers and products
customer_ids=["customer_"+str(1+i) for i in range(100)]
customer_data = np.random.random((100,32))
product_ids=["product_"+str(101+i) for i in range(100)]
product_data = np.random.random((100,32))
sim.add_items(customer_data,customer_ids,partition="customers")
sim.add_items(product_data,product_ids,partition="products")
# Index the data
sim.init()
# Run nearest neighbor vector search
query = np.random.random(32)
dists, items = self.sim.search(query,k=10,partition="customers") # returns a list of customers and products
dists, items = self.sim.search(query,k=10) # returns a list of only customers
```

For more examples, please read our [documentation](https://vecsim.readthedocs.io/)
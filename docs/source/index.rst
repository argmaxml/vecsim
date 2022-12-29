VecSim
==========================================
VecSim is a lightweight wrapper that provides a standard interface for similarity search over vectors.

Why use VecSim
-------------------------------------------

1. **Standard API** - Different vector similarity servers have different APIs - so switching is not trivial.
2. **Identifiers** - Some vector similarity servers support string IDs, some do not - we keep track of the mapping.
3. **Partitions** - In most cases, pre-filtering is needed prior to querying, we abstract this concept away.

Installation
---------------

First, install the package:

```
pip install vecsim
```

If you are using a specific similarity engine, such as redis or faiss - you can install it as well:

```
pip install vecsim[redis]
pip install vecsim[faiss]
``` 


Then, you can use the package as follows:

.. code-block:: python

    from vecsim import SciKitIndex

Choose the similarity engine, see the :doc:`Supported Engines<engines>` section for more details.

Quick Start
----------------

.. code-block:: python

    import numpy as np
    from vecsim import SciKitIndex
    sim = SciKitIndex(metric='cosine', dim=32)

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
    dists, items = sim.search(query,k=10,partition="users") # returns a list of users only


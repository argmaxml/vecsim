Supported Similarity Engines
==========================================

The following similarity engines are supported:

1. Scikit-learn
2. Faiss
3. Redis
4. Elasticsearch
5. Pinecone

Scikit-learn
----------------

Scikit-learn is a Python module for machine learning built on top of SciPy. It
provides a range of supervised and unsupervised learning algorithms via a
consistent interface in Python. It is a good choice for small datasets.


**Example:**

.. code-block:: python

    from vecsim import SciKitIndex
    sim = SciKitIndex(metric='cosine', dim=32)

    # Add data to the index
    sim.add_items(user_data, user_ids, partition="users") # Will store a copy of the data for the "users" partition in memory
    sim.add_items(item_data, item_ids, partition="items") # Will store a copy of the data for the "items" partition in memory

    # Index the data
    sim.init() # This will invoke the `fit` method of the underlying scikit-learn model



Faiss
----------------

Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size, up
to ones that possibly do not fit in RAM. It also contains supporting code for
evaluation and parameter tuning. Faiss is written in C++ with complete
wrappers for Python/numpy. Some of the most useful algorithms are implemented
on the GPU. It is a good choice for medium to large datasets.


The implementation of the FaissIndex class is based on the `faiss` Python package.
We use the `index_factory` parameter to create the index. The `index_factory`
parameter is a string that describes the type of the index to create, possibly
followed by a comma-separated list of options. The following index types are
supported:

- Flat: brute-force search
- IVF, IVFADC, IVFPQ: inverted file methods
- HNSW: hierarchical navigable small world

For more information about the index types, please refer to the `faiss` documentation.

https://github.com/facebookresearch/faiss/wiki/The-index-factory

**Example:**

.. code-block:: python

    from vecsim import FaissIndex
    sim = FaissIndex(metric='cosine', dim=32, index_factory="Flat")

    # Add data to the index
    sim.add_items(user_data, user_ids, partition="users")
    sim.add_items(item_data, item_ids, partition="items") 

    # Index the data
    sim.init() # Will create one instance of faiss per partition.

Redis
----------------

Redis is an open source (BSD licensed), in-memory data structure store, used
as a database, cache and message broker. It supports data structures such as
strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs,
geospatial indexes with radius queries and streams. It is a good choice for
small to medium datasets.

**Example:**

.. code-block:: python

    from vecsim import RedisIndex
    sim = RedisIndex(metric='cosine', dim=32, redis_credentials={"host":"127.0.0.1", "port": 6379}, overwrite=True)

    # Add data to the index
    sim.add_items(user_data, user_ids, partition="users")
    sim.add_items(item_data, item_ids, partition="items") 

    # Index the data
    sim.init()

Elasticsearch
----------------

Elasticsearch is a distributed, RESTful search and analytics engine capable of
solving a growing number of use cases. Elasticsearch is built on Apache Lucene
that provides a full-text search engine with vector similarity search and
geo-spatial capabilities. Elasticsearch is a good choice for medium to large
datasets. Elasticsearch is the preferred choice when the majority of the vectors are sparse.


**Example:**

.. code-block:: python

    from vecsim import ElasticIndex
    sim = ElasticIndex(metric='cosine', dim=32, elastic_credentials={"hosts": "http://127.0.0.1:9200"})


Pinecone
----------------

Elasticsearch is a distributed, RESTful search and analytics engine capable of
solving a growing number of use cases. Elasticsearch is built on Apache Lucene
that provides a full-text search engine with vector similarity search and
geo-spatial capabilities. Elasticsearch is a good choice for medium to large
datasets. Elasticsearch is the preferred choice when the majority of the vectors are sparse.


**Example:**

.. code-block:: python

    from vecsim import PineconeIndex
    sim = PineconeIndex(metric='cosine', dim=32,pinecone_credentials={"api_key": "PINECONE_API_KEY", "environment": "PINECONE_ENV"})

import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from vecsim import RedisIndex

# Requires: a running docker on the local host
#          docker run -p 6379:6379 -d redis

class RedisTest(unittest.TestCase):
    def setUp(self):
        self.sim = RedisIndex(metric='cosine',dim=32,redis_credentials={"host":"127.0.0.1", "port": 6379}, overwrite=True)
    
    def test_search(self):
        data=np.eye(32)
        aids=["a"+str(1+i) for i in range(32)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition=None)
        self.assertEqual(len(items),10)
        self.assertEqual(dists[0],0)
        self.assertEqual(items[0],"a1")

    def test_partition(self):
        data=np.random.random((100,32))
        aids=["a"+str(1+i) for i in range(100)]
        bids=["b"+str(101+i) for i in range(100)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.add_items(data,bids,partition="b")
        self.sim.init()
        dists, items = self.sim.search([data[0]],k=10,partition="b")
        self.assertTrue(all([i.startswith("b") for i in items]))
    
        



if __name__ == '__main__':
    unittest.main()
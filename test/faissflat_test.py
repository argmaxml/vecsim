import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from vecsim import FaissFlatIndex, FaissIVFPQIndex


class FlatFaissTest(unittest.TestCase):
    
    def test_search(self):
        self.sim = FaissFlatIndex(metric='cosine',dim=32)
        data=np.eye(32)
        aids=["a"+str(1+i) for i in range(32)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition=None)
        self.assertEqual(items[0],"a1")

    def test_partition(self):
        self.sim = FaissFlatIndex(metric='cosine',dim=32)
        data=np.random.random((100,32))
        aids=["a"+str(1+i) for i in range(100)]
        bids=["b"+str(101+i) for i in range(100)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.add_items(data,bids,partition="b")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition="b")
        self.assertTrue(all([i.startswith("b") for i in items]))
    
    def test_search_ivf(self):
        self.sim = FaissIVFPQIndex(metric='cosine',dim=32)
        n= 10000
        data=np.random.random((n,32))
        aids=["a"+str(1+i) for i in range(n)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition=None)
        self.assertEqual(items[0],"a1")

        



if __name__ == '__main__':
    unittest.main()
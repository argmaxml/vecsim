import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from vecsim import SciKitIndex


class SciKitTest(unittest.TestCase):
    def setUp(self):
        self.sim = SciKitIndex(metric='cosine',dim=32)
    
    def test_search(self):
        data=np.eye(32)
        aids=["a"+str(1+i) for i in range(32)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition=None)
        self.assertEqual(len(items),10)
        self.assertEqual(dists[0],0)
        self.assertEqual(items[0],"a1")

    def test_search_agg_mean(self):
        n = 32
        data=np.eye(n)+np.hstack([np.eye(n),np.zeros((n,1))])[:,1:]
        data[0,-1] = 1
        aids=["a"+str(i//2)+"_"+str(i%2) for i in range(n)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.init()
        dists, items = self.sim.search_and_aggregate(data[1],k=10,partition=None, aggfunc='mean', delim="_")
        self.assertEqual(items[0],"a0")
        self.assertEqual(items[1],"a1")
        self.assertAlmostEqual(dists[0],0.25)
        self.assertAlmostEqual(dists[1],0.5)

    def test_partition(self):
        data=np.random.random((100,32))
        aids=["a"+str(1+i) for i in range(100)]
        bids=["b"+str(101+i) for i in range(100)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.add_items(data,bids,partition="b")
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition="b")
        self.assertTrue(all([i.startswith("b") for i in items]))
    
    def test_partition_list(self):
        data=np.random.random((200,32))
        ids=["a"+str(1+i) for i in range(100)]+["b"+str(101+i) for i in range(100)]
        ps = np.array(["a"]*100 + ["b"]*100)
        self.sim.add_items(data,ids,partition=ps)
        self.sim.init()
        dists, items = self.sim.search(data[0],k=10,partition="b")
        self.assertTrue(all([i.startswith("b") for i in items]))



if __name__ == '__main__':
    unittest.main()
import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from vecsim.similarity_helpers import PineconeIndex

class PineconeTest(unittest.TestCase):
    def setUp(self):
        self.sim = PineconeIndex(metric='cosine',dim=32, pinecone_credentials={"api_key": "2b8090ad-e880-48ef-bd4a-1c14c5af5c0f", "environment": "us-west4-gcp"})
    
    def test_search(self):
        data=np.eye(32)
        aids=["a"+str(1+i) for i in range(32)]
        self.sim.add_items(data,aids,partition="a")
        dists, items = self.sim.search(data[0],k=10,partition=None)
        self.assertEqual(len(items),10)
        self.assertEqual(dists[0],1)
        self.assertEqual(items[0],"a1")

    def test_partition(self):
        data=np.random.random((100,32))
        aids=["a"+str(1+i) for i in range(100)]
        bids=["b"+str(101+i) for i in range(100)]
        self.sim.add_items(data,aids,partition="a")
        self.sim.add_items(data,bids,partition="b")
        dists, items = self.sim.search(data[0],k=10,partition="b")
        self.assertTrue(all([i.startswith("b") for i in items]))
    
        
if __name__ == '__main__':
    unittest.main()
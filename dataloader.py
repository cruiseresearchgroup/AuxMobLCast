from torch.utils.data import DataLoader
from dataset import MobilityDataset
import numpy as np

class MobilityDataLoader(DataLoader):
    def __init__(self, inp, out, shuf_out, enc_tokenizer, dec_tokenizer, num_workers = 6, batch_size = 128, pin_memory = True, 
                 shuffle = True):
        self.inp = inp
        self.out = out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.pin_memory = pin_memory
        self.shuf_out = shuf_out

        self.dataset = MobilityDataset(self.inp, self.out, self.shuf_out, self.enc_tokenizer, self.dec_tokenizer, None)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)
    
    @staticmethod
    def collate_fn(data):
        inp, out, shuf_out, inp_mask, out_mask = zip(*data)
        
        labels = np.array(out)
        
        for i, report_mask in enumerate(out_mask):
          for k, j in enumerate(report_mask):
            if j == 1:
              pass
            else:
              labels[i, k] = -100
              
        return np.array(inp), np.array(out), np.array(shuf_out, dtype = np.int64), np.array(inp_mask), np.array(out_mask), np.array(labels)
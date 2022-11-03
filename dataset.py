from torch.utils.data import Dataset

class MobilityDataset(Dataset):
  def __init__(self, inp_data, out_data, shuf_out_data, enc_tokenizer, dec_tokenizer, transforms = None):
      
    self.inp_data = inp_data
    self.out_data = out_data
    self.shuf_out_data = shuf_out_data
    self.enc_tokenizer = enc_tokenizer
    self.dec_tokenizer = dec_tokenizer

  def __len__(self):
        return len(self.inp_data)

  def __getitem__(self, idx):
        
    encoded_inp = self.enc_tokenizer(self.inp_data[idx], add_special_tokens = True, max_length = 60,
                                return_attention_mask = True, truncation = True, padding = "max_length")
    
    encoded_out = self.dec_tokenizer(self.dec_tokenizer.eos_token + self.out_data[idx].split(' ')[3], add_special_tokens = True, 
                                     max_length = 2, return_attention_mask = True, truncation = True, padding = "max_length")
    
    shuf_encoded_inp = self.enc_tokenizer(self.shuf_out_data[idx], add_special_tokens = False, 
                                     max_length = 1, return_attention_mask = True, truncation = True, padding = "max_length")
    
    shuf_encoded_out = self.enc_tokenizer.batch_decode([shuf_encoded_inp['input_ids']], skip_special_tokens = True)
    
    return encoded_inp['input_ids'], encoded_out['input_ids'], shuf_encoded_out, encoded_inp['attention_mask'], encoded_out['attention_mask']
import os
import subprocess
from transformers import EncoderDecoderModel, BertTokenizerFast, LineByLineTextDataset, Trainer, TrainingArguments, BertTokenizer, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import Trainer
import time
import traceback
import json
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from torch.cuda.amp import autocast
from transformers import BartForConditionalGeneration, BertForSequenceClassification, BartTokenizer, AutoModel, AutoModelForCausalLM, T5TokenizerFast, \
T5ForConditionalGeneration, BertForPreTraining, DataCollatorForLanguageModeling, BertForMaskedLM, XLNetTokenizer, XLNetTokenizer, \
XLNetLMHeadModel, DistilBertTokenizerFast, RobertaTokenizer, RobertaModel, RobertaForCausalLM, BertModel, BertLMHeadModel, \
GPT2Tokenizer, DistilBertTokenizer
import logging
from dataloader import MobilityDataLoader
from metrics import np_evaluate
from dataset import MobilityDataset
from transformers import EarlyStoppingCallback
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import random
from encoder_decoder import EncoderDecoderModel
from bert_encoder import BertModel
import argparse

def build_optimizer(model):
    
    # optimizer = getattr(torch.optim, 'Adam')(
    #     [{'params': model.encoder.parameters(), 'lr': 5e-6},
    #      {'params': model.decoder.parameters(), 'lr': 5e-5}],
    #     weight_decay = 5e-4,
    #     amsgrad = True
    # )
    optimizer = getattr(torch.optim, 'Adam')(
        [{'params': model.parameters(), 'lr': 5e-5}],
        weight_decay = 5e-4,
        amsgrad = True
    )
    return optimizer


def build_lr_scheduler(optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, 'ReduceLROnPlateau')(optimizer, mode = 'min', patience = 6, cooldown = 2,
                                                                          verbose = True)
    # lr_scheduler = getattr(torch.optim.lr_scheduler, 'StepLR')(optimizer, step_size = 25, verbose = False)
    return lr_scheduler
def eval_model(logger, model, epoch_time, tokenizer, optimizer, epoch, dataloader, model_name, device):
    
    model.eval()
    with torch.no_grad():
        val_gts, val_res = [], []   
        running_loss_eval = 0.0
        running_loss_clip = 0.0
        for i, (inp, out, shuf_labels, enc_mask, dec_mask, labels) in enumerate(dataloader):
    
            inp = torch.LongTensor(inp).to(device)
            out = torch.LongTensor(out).to(device)
            shuf_labels = torch.LongTensor(shuf_labels).to(device)
            enc_mask = torch.LongTensor(enc_mask).to(device)
            dec_mask = torch.LongTensor(dec_mask).to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()

            output = model.generate(input_ids = inp, do_sample = True, decoder_start_token_id = tokenizer.eos_token_id, 
                                    top_k = 40, max_length = 2, \
                                    early_stopping = True, num_beams = 2)
            
            reports = tokenizer.batch_decode(output, skip_special_tokens = True)
            ground_truths = tokenizer.batch_decode(out, skip_special_tokens = True)
            val_res.extend(reports)
            val_gts.extend(ground_truths)
    
    print(f"Time taken for val loss: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
    logger.info(f"Time taken generation val: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
    
    gen_values = {}
    gts = []
    res = []
    try:
        for i in range(0, len(val_res)):
            gen_values[i] = {'gts' : val_gts[i],
                            'res' : val_res[i]}
            gts.append(int(val_gts[i]))
            res.append(int(val_res[i]))
        
        if not os.path.exists(f"{model_name}"):
            os.makedirs(f"{model_name}")
            
        filename = f"{model_name}/val_{model_name}_{epoch}.json"
        with open(filename, "w") as write_file:
            json.dump(gen_values, write_file, indent=4)
        
        rmse, mae = np_evaluate(gts, res)
        
        print(f"The val metrics for epoch {epoch}: RMSE: {rmse:.3f} MAE: {mae:.3f}")
        logger.info(f"The val metrics for epoch {epoch}: RMSE: {rmse:.3f} MAE: {mae:.3f}")
    except Exception as e:
        logging.error(traceback.format_exc())
        rmse = 100
        mae = 100
        pass

    print(f"Time taken for metrics calculation val: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
    logger.info(f"Time taken for metrics calculation val: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
    print()
    # return rmse, mae, epoch_loss_eval, epoch_loss_clip
    return rmse, mae
    
def train_model(logger, model, tokenizer, criterion, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, 
                model_name, scaler = None, num_epochs = 70, stop = 10, device='cuda', checkpoint = None):
    
    since = time.time()
    best_loss = 100000
    best_rmse = 1e5
    best_mae = 1e5
    
    if checkpoint != None:
        print('Checkpoint found')
        logger.info('Checkpoint found')
        start_epoch = checkpoint['epochs'] + 1
        print(f"Resuming training from {start_epoch}")
        logger.info(f"Resuming training from {start_epoch}")
        del checkpoint
    else:
        start_epoch = 1
        print('------------TRAINING STARTED---------------------')
        logger.info('------------TRAINING STARTED---------------------')
    
    early_stopping = 0
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        logger.info('-' * 10)

        running_loss = 0.0
        total = 0
        correct = 0
        
        # Iterate over data.
        for i, (inp, out, shuf_labels, enc_mask, dec_mask, labels) in enumerate(train_dataloader):

            inp = torch.LongTensor(inp).to(device)
            out = torch.LongTensor(out).to(device)
            shuf_labels = torch.LongTensor(shuf_labels).to(device)
            enc_mask = torch.LongTensor(enc_mask).to(device)
            dec_mask = torch.LongTensor(dec_mask).to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()
            
            with autocast():
                output = model(input_ids = inp, attention_mask = enc_mask, decoder_input_ids = out, decoder_attention_mask = dec_mask, 
                               labels = labels, shuf_labels = shuf_labels)
                loss_ce, loss_clip = output.loss
                loss = 0.8 * loss_ce + 0.2 * loss_clip
            del inp, out, enc_mask, labels
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)        
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'scheduler' : scheduler.state_dict(),
                      'epochs' : epoch,
                      'early_stopping' : early_stopping}

        print('Total Train Loss: {:.4f}'.format(epoch_loss))
        logger.info('Total Train Loss: {:.4f}'.format(epoch_loss))
        
        print(f"Time taken for training epoch {epoch}: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
        logger.info(f"Time taken for training epoch {epoch}: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
        
        rmse, mae, eval_loss = eval_model(logger, model, epoch_time, tokenizer, optimizer, epoch, val_dataloader, model_name, device)
        if rmse < best_rmse:
            early_stopping = 0
            best_rmse = rmse 
            logger.info(best_rmse)
            checkpoint = {'state_dict': model.state_dict(),
                          'epochs' : epoch,
                          'best_rmse' : best_rmse,
                          'early_stopping' : early_stopping}

            torch.save(checkpoint, f"bmobility_{model_name}.pth")
            logger.info('Best checkpoint saved (RMSE)')
        elif mae < best_mae:
            early_stopping = 0
            best_mae = mae 
            logger.info(best_mae)
            checkpoint = {'state_dict': model.state_dict(),
                          'epochs' : epoch,
                          'best_mae' : best_mae,
                          'early_stopping' : early_stopping}

            torch.save(checkpoint, f"bmobility_mae_{model_name}.pth")
            logger.info('Best checkpoint saved (MAE)')
        else:
            early_stopping += 1
        if early_stopping == stop:
            print('Stopping early since RMSE is not improving')
            logger.info(f"Stopping early since RMSE is not improving")
            break
        
        scheduler.step(rmse)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--seed', type=int)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    

    logging.basicConfig(filename=f"{args.log_dir}",
                format='%(asctime)s %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S",
                filemode='a')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    with open(f"{args.dataset_type}/RMPOI_train_input.txt", 'r') as f:
        train_input = f.read().split('\n')

    with open(f"{args.dataset_type}/train_output.txt", 'r') as f:
        train_output = f.read().split('\n')
    
    with open(f"{args.dataset_type}/POI_train_CLS.txt", 'r') as f:
        shuf_train_output = f.read().split('\n')

    with open(f"{args.dataset_type}/RMPOI_test_input.txt", 'r') as f:
        test_input = f.read().split('\n')

    with open(f"{args.dataset_type}/test_output.txt", 'r') as f:
        test_output = f.read().split('\n')
        
    with open(f"{args.dataset_type}/POI_test_CLS.txt", 'r') as f:
        shuf_test_output = f.read().split('\n')

    with open(f"{args.dataset_type}/RMPOI_val_input.txt", 'r') as f:
        val_input = f.read().split('\n')
    
    with open(f"{args.dataset_type}/val_output.txt", 'r') as f:
        val_output = f.read().split('\n')
    
    with open(f"{args.dataset_type}/POI_val_CLS.txt", 'r') as f:
        shuf_val_output = f.read().split('\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = args.model_name
    
    logger.info(model_name)

    
    # enc_tokenizer = BertTokenizer.from_pretrained("../pretrained_models/bert-base-uncased/")
    enc_tokenizer = RobertaTokenizer.from_pretrained("../pretrained_models/roberta-base/")
    # enc_tokenizer = XLNetTokenizer.from_pretrained("../pretrained_models/xlnet-base-cased/")
    
    dec_tokenizer = GPT2Tokenizer.from_pretrained("../pretrained_models/gpt2")
    dec_tokenizer.pad_token = dec_tokenizer.eos_token

    
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained('../pretrained_models/bert-base-uncased/',
    #                                                             '../pretrained_models/gpt2')
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained('../pretrained_models/xlnet-base-cased/',
    #                                                             '../pretrained_models/gpt2')
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('../pretrained_models/roberta-base/',
                                                                '../pretrained_models/gpt2')  
    
        
    model.load_state_dict(torch.load(f"bmobility_{args.model_name}.pth")["state_dict"],
                          strict = False)
    
    
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(train_params)
    non_train_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(non_train_params)
    
    model.to(device)

    train_dataloader = MobilityDataLoader(train_input, train_output, shuf_train_output, enc_tokenizer, dec_tokenizer, shuffle = True, 
                                          pin_memory = True)
    val_dataloader = MobilityDataLoader(val_input, val_output, shuf_val_output, enc_tokenizer, dec_tokenizer, shuffle = False, 
                                        pin_memory = True)
    test_dataloader = MobilityDataLoader(test_input, test_output, shuf_test_output, enc_tokenizer, dec_tokenizer, shuffle = False, 
                                         pin_memory = True)
    
    criterion = nn.CrossEntropyLoss()

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)
    
    rmse, mae = eval_model(logger, model, time.time(), dec_tokenizer, optimizer, 1, test_dataloader, model_name, device)
    
    logger.info(rmse)
    logger.info(mae)
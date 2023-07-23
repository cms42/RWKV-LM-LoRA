########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print("Loading libraries...")
import numpy as np
import json
from tqdm import tqdm
from rwkv import rwkv_tokenizer

print("Loading tokenizer...")
tokenizer = rwkv_tokenizer.TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

input_file = "train-v3.jsonl"
output_file = 'train-v3.npz'

ctx_len = 2049
allow_trunc = False
dels = []

TASK = 'tokenize' # tokenize

if TASK == 'tokenize':
    print("Reading input file...")
    with open(input_file, "r", encoding="utf-8") as f:
        f = f.readlines()
        l = len(f)
        tokens = np.ndarray(shape=(l,ctx_len),dtype='int32')
        lens = np.ndarray(shape=(l,2),dtype='int32')
        tokens.fill(0)
        print("Tokenizing...")
        for i,line in tqdm(enumerate(f)):
            item = json.loads(line)
            t = tokenizer.encode(item['prompt'])
            t1 = tokenizer.encode(item['text'])
            lens[i,0] = len(t)
            lens[i,1] = len(t1)
            if(lens[i,0]>=ctx_len or lens[i,0]+lens[i,1]>ctx_len and not allow_trunc):
                dels.append(i)
                continue
            tokens[i,0:lens[i,0]] = t
            tokens[i,lens[i,0]:lens[i,0]+lens[i,1]] = t1[0:ctx_len-lens[i,0]]
    if(len(dels)>0):
        print(f"Deleting unacceptably LONG lines{dels}...")
        tokens = np.delete(tokens, dels, 0)
        lens = np.delete(lens, dels, 0)
        print(f"Now its shape is {tokens.shape}")
    print("Writing output...")
    np.savez_compressed(output_file, tokens=tokens, lens=lens)
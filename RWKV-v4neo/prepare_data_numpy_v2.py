########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import jsonlines
from tqdm import tqdm

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

input_file = r"E:\LLM\Mo\train.jsonl"
output_file = 'train.npy'

TASK = 'tokenize' # tokenize verify

if TASK == 'tokenize':
    a1 = []
    a2 = []
    with open(input_file, "r+", encoding="utf8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            t = tokenizer.encode(item['prompt'])
            t1 = tokenizer.encode(item['text']+'<|endoftext|>')
            a1.append(np.array(t+t1,dtype='int32'))
            a2.append(np.array([0]*len(t)+[1]*len(t1),dtype='uint8'))
    o1 = np.ndarray(shape=(len(a1),2),dtype='object')
    o1[:,0] = a1
    o1[:,1] = a2
    np.save(output_file, o1, allow_pickle=True)
    

elif TASK == 'verify':

    test = np.load(output_file)
    print(test)
    print('\n\n')
    print(tokenizer.decode(test[:100]))
    print('\n\n')
    print(tokenizer.decode(test[-100:]))
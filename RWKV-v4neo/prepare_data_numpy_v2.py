print("Loading libraries...")
import numpy as np
import json
from tqdm import tqdm
from rwkv import rwkv_tokenizer

print("Loading tokenizer...")
tokenizer = rwkv_tokenizer.TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

input_file = "train-v2.jsonl"
output_file = 'train-v2.npy'

TASK = 'tokenize' # tokenize verify

if TASK == 'tokenize':
    a1 = []
    a2 = []
    print("Reading input file...")
    with open(input_file, "r", encoding="utf-8") as f:
        f = f.readlines()
        print("Tokenizing...")
        for line in tqdm(f):
            item = json.loads(line)
            t = tokenizer.encode(item['prompt'])
            t1 = tokenizer.encode(item['text'])
            a1.append(np.array(t+t1+[0],dtype='int32'))
            a2.append(np.array([0]*len(t)+[1]*len(t1)+[0],dtype='uint8'))
    print("Writing output...")
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
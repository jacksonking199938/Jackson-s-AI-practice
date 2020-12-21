#%%
from d2l import torch as d2l
import math
import torch
import os
import random

#%%
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                        '319d85e578af0cdc590547f26231e4e31cdf1e42')
def read_ptb():
    data_dir = d2l.download_extract('ptb')
    with open(os.path.join(data_dir,'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'

#%%
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size:{len(vocab)}'
# %%
def subsampling(sentences, vocab):
    # Map low frequency words into <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
                    for line in sentences]
    # Count the frequency for each word
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return (random.uniform(0,1) < 
                math.sqrt(1e-4 / counter[token] * num_tokens))
    
    return [[tk for tk in line if keep(tk)] for line in sentences]

subsampled = subsampling(sentences, vocab)

#%%
d2l.set_figsize()
d2l.plt.hist([[len(line) for line in sentences],
                [len(line) for line in subsampled]])
d2l.plt.xlabel('# tokens per sentence')
d2l.plt.ylabel('count')
d2l.plt.legend(['origin','subsampled'])


# %%
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([line.count(token) for line in sentences])}, '
            f'after={sum([line.count(token) for line in subsampled])}')
#%%
compare_counts('the')

# compare_counts('join')
#%%
corpus = [vocab[line] for line in subsampled]
corpus[0:3]

# %%
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0,i-max_window_size),
                                min(len(line), i+1+window_size)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
#%%
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
# %%
class RandomGenerator:
    """Draw a random int in [0,n] according to n sampling weights"""
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i -1]
    
# %%
def get_negatives(all_contexts, corpus, K):
    counter = d2l.count_corpus(corpus)
    sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, corpus,5)

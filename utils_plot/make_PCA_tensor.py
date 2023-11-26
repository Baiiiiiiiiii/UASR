import os
import librosa
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import textgrid
import torchaudio
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2Model
import seaborn as sns
from tqdm import tqdm
import torch.multiprocessing as mp
# from tsnecuda import TSNE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_textgrid(textgrid_file):
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        # print(tg.tiers[1])
        phone_tier = tg.tiers[1]
        intervals = phone_tier.intervals
        phoneme_timestamps = []

        for interval in intervals:
            # print(interval)
            start_time = interval.minTime
            end_time = interval.maxTime
            phoneme = interval.mark
            phoneme_timestamps.append((start_time, end_time, phoneme))

        return phoneme_timestamps # return tuples in a list

    except Exception as e:
        print(f"Error parsing TextGrid file {textgrid_file}: {e}")
        return []




X = torch.load('10000_X_tensor.pt')
phonemes = torch.load('10000_phonemes_tensor.pt')
# Step 4: PCA Transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  
torch.save(X_pca, '10000_pca.pt')


X = torch.load('1000_X_tensor.pt')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) 
torch.save(X_pca, '1000_pca.pt')





# X = torch.load('10000_X_tensor.pt')
# phonemes = torch.load('10000_phonemes_tensor.pt')
# print(X)
# # Step 4: t-SNE Transformation
# tsne = TSNE(n_components=2, perplexity=50,learning_rate=10)
# X_tsne = tsne.fit_transform(X)
# X_tsne = X_tsne.cpu()  
# torch.save(X_tsne, 'tsne_ppl_50.pt')

# tsne = TSNE(n_components=2, random_state=42, perplexity=50)
# X_tsne = tsne.fit_transform(X)  
# torch.save(X_tsne, 'tsne_ppl_50_val.pt')
# torch.save(phonemes, 'phonemes_ppl_30_val.pt')
# print(X_tsne) 



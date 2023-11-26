import os
import librosa
import numpy as np
from sklearn.manifold import TSNE
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
# Step 4: t-SNE Transformation
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)  
torch.save(X_tsne, 'tsne_ppl_30.pt')
                    
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


# draw plot
# load torch files
perplexity = 10
# X_tsne = torch.load(f"tsne_ppl_{perplexity}_val.pt")
# phonemes = torch.load('phonemes_list.pt')


df = pd.DataFrame()
df["phonemes"] = phonemes
df["comp-1"] = X_tsne[:,0]
df["comp-2"] = X_tsne[:,1]



df_length = df.shape
print("df_length: ", df_length)
plt.figure(figsize=(26, 24))
ax = sns.scatterplot(x="comp-1",y="comp-2", hue=df.phonemes.tolist(), data=df)
legend = ax.legend()
for text in legend.get_texts():
    text.set_fontsize('14')  # Adjust the legend text font size
# save the plot as PNG file
plt.savefig("tsne_ppl_30.png")
# Clear the current figure
plt.clf()




# Find the top 10 most frequent labels
# top_10_labels = df["phonemes"].value_counts().nlargest(10).index.tolist()
# # print(top_10_labels)

# # Filter the DataFrame to include only the top 10 labels
# filtered_df = df[df["phonemes"].isin(top_10_labels)]
# filtered_df_length = filtered_df.shape
# print("filtered_df_length: ", filtered_df_length)
# # Create the scatterplot with the "tab20" colormap
# ax = sns.scatterplot(x="comp-1", y="comp-2", hue=filtered_df.phonemes.tolist(), data=filtered_df)
# # Adjust the legend size
# legend = ax.legend()
# for text in legend.get_texts():
#     text.set_fontsize('12')  # Adjust the legend text font size
# # Save the scatterplot figure
# plt.savefig("tsne_ppl_30_val_label_10.png")
# # Clear the current figure
# plt.clf()




desired_labels = [
    'AA0', 'AA1', 'AA2',
    'AE0', 'AE1', 'AE2',
    'AH0', 'AH1', 'AH2',
    'AO0', 'AO1', 'AO2',
    'AY0', 'AY1', 'AY2',
    'EY0', 'EY1', 'EY2',
    'IY0', 'IY1', 'IY2',
    'OW0', 'OW1', 'OW2',
    'UH0', 'UH1', 'UH2',
    'UW0', 'UW1', 'UW2'
]

desired_labels = [
    ['AA0', 'AA1', 'AA2',],
    ['AE0', 'AE1', 'AE2',],
    ['AH0', 'AH1', 'AH2',],
    ['AO0', 'AO1', 'AO2',],
    ['AW0', 'AW1', 'AW2'],
    ['AY0', 'AY1', 'AY2',],
    ['EH0', 'EH1', 'EH2'],
    ['ER0', 'ER1', 'ER2'],
    ['EY0', 'EY1', 'EY2',],
    ['IH0', 'IH1', 'IH2',],
    ['IY0', 'IY1', 'IY2',],
    ['OW0', 'OW1', 'OW2',],
    ['OY0', 'OY1', 'OY2'],
    ['UH0', 'UH1', 'UH2',],
    ['UW0', 'UW1', 'UW2',],
]



# vowel_df = df[df["phonemes"].isin(desired_labels)]

# vowel_df_length = vowel_df.shape
# print("vowel_df_length: ", vowel_df_length)
# # Set the figure size
# plt.figure(figsize=(25, 23))
# # Create the scatterplot with the "tab20" colormap , palette="tab20"
# ax = sns.scatterplot(x="comp-1", y="comp-2", hue=vowel_df.phonemes.tolist(), data=vowel_df)
# # Adjust the legend size
# legend = ax.legend()
# for text in legend.get_texts():
#     text.set_fontsize('12')  # Adjust the legend text font size
# # Save the scatterplot figure
# plt.savefig(f"tsne_ppl_{perplexity}_val_label_vowel.png")
# plt.clf()


# for phonemes in desired_labels:
#     vowel_df = df[df["phonemes"].isin(phonemes)]
#     vowel_df_length = vowel_df.shape
#     print("vowel_df_length: ", vowel_df_length)
#     # Set the figure size
#     plt.figure(figsize=(10, 10))
#     # Create the scatterplot with the "tab20" colormap , palette="tab20"
#     ax = sns.scatterplot(x="comp-1", y="comp-2", hue=vowel_df.phonemes.tolist(), data=vowel_df)
#     # Adjust the legend size
#     legend = ax.legend()
#     for text in legend.get_texts():
#         text.set_fontsize('12')  # Adjust the legend text font size
#     # Save the scatterplot figure
#     plt.savefig(f"tsne_ppl_{perplexity}_val_{phonemes[0][:-1]}_vowel.png")
#     plt.clf()






# ==============combine vowel==========================
# Create a mapping of labels to their categories
# label_to_category = {
#     'AA0': 'AA', 'AA1': 'AA', 'AA2': 'AA',
#     'AE0': 'AE', 'AE1': 'AE', 'AE2': 'AE',
#     'AH0': 'AH', 'AH1': 'AH', 'AH2': 'AH',
#     'AO0': 'AO', 'AO1': 'AO', 'AO2': 'AO',
#     'AY0': 'AY', 'AY1': 'AY', 'AY2': 'AY',
#     'EY0': 'EY', 'EY1': 'EY', 'EY2': 'EY',
#     'IY0': 'IY', 'IY1': 'IY', 'IY2': 'IY',
#     'OW0': 'OW', 'OW1': 'OW', 'OW2': 'OW',
#     'UH0': 'UH', 'UH1': 'UH', 'UH2': 'UH',
#     'UW0': 'UW', 'UW1': 'UW', 'UW2': 'UW',
# }

# # Add a new column to your DataFrame with categories
# df["category"] = df["phonemes"].map(label_to_category)

# # Set the figure size
# plt.figure(figsize=(10, 10))

# # Create the scatterplot with the "tab20" colormap using the "category" column
# ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.category.tolist(), data=df)

# # Adjust the legend size
# legend = ax.legend()
# for text in legend.get_texts():
#     text.set_fontsize('12')  # Adjust the legend text font size

# # Save the scatterplot figure
# plt.savefig(f"tsne_ppl_{perplexity}_val_label_vowel_combined.png")


























# fig, axes = plt.subplots(1, 1, figsize=(20, 20))
# all_feature = TSNE(2, init='pca', learning_rate='auto').fit_transform(X)
# scatter = axes[0].scatter(all_feature[..., 0], all_feature[..., 1],c=all_label, alpha=0.5, s=10)
# axes[0].legend(*scatter.legend_elements(), title='phoneme')
# axes[0].set_title("wav2veca representation")
# fig.savefig('tsne_phoneme_representations_test.png')

















# # Step 2: Feature Extraction
# def extract_features(wav_file):
#     y, sr = librosa.load(wav_file, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     return mfcc.mean(axis=1)  # Use the mean of MFCCs as the feature vector

# # Extract features and phonemes
# X = []
# phonemes = []
# wavs_dir = './Data/LJSpeech-1.1/wavs/'
# for wav_file in os.listdir(wavs_dir):
#     if wav_file.endswith('.wav'):
#         wav_path = os.path.join(wavs_dir, wav_file)
#         wav_name = os.path.splitext(wav_file)[0]
#         if wav_name in mapping:
#             X.append(extract_features(wav_path))
#             phonemes.append(mapping[wav_name])

# # Step 3: t-SNE Transformation
# X = np.array(X)
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # Step 4: Visualization
# df = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2'])
# df['Phonemes'] = phonemes

# # Create a scatter plot with colors based on phonemes
# plt.figure(figsize=(12, 8))
# colors = plt.cm.viridis(np.linspace(0, 1, len(df['Phonemes'].unique())))
# for i, (phoneme, color) in enumerate(zip(df['Phonemes'].unique(), colors)):
#     subset = df[df['Phonemes'] == phoneme]
#     plt.scatter(subset['Component 1'], subset['Component 2'], c=[color], label=phoneme)

# plt.legend(title='Phonemes')
# plt.title('t-SNE Visualization of LJSpeech Phonemes')
# # Save the figure as an image
# plt.savefig('tsne_visualization.png')
# plt.show()


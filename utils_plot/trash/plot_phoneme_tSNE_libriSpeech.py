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

device = 'cuda'

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



# Step 1: Data Preparation
# Load the mapping between waveform names and phoneme transcriptions
mapping = {}
count = 0 
# with open('./preprocessed_data/LJSpeech/train.txt') as f:
# with open('./preprocessed_data/LJSpeech/val.txt') as f:    
#     for line in f:
#         # if count <10:
#         parts = line.strip().split('|')
#         mapping[parts[0]] = parts[2]
#         # count +=1
# print(mapping)


# Define the directory containing the .txt files
directory = '../LibriSpeech/train-clean-360/' # raw wave dir
# Initialize an empty list to store wave names
wave_names = []
wave_paths = []
# Loop through all subdirectories and files in the specified directory
for dir_path, subdirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a .txt file
        if file.endswith('.trans.txt'):
            file_path = os.path.join(dir_path, file)
            
            # Read lines from the .txt file
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Extract wave names and add them to the list
                for line in lines:
                    wave_name = line.split()[0]
                    wave_names.append(wave_name)
                    wave_paths.append(os.path.join(dir_path, wave_name+'.flac'))

# Print the list of wave names
# print(wave_paths)
# breakpoint()


textgrid_root_dir = '../librispeech_alignments/train-clean-360'
# Initialize an empty list to store textgrid names
textgrid_names = []
textgrid_filepaths = []
# Loop through all subdirectories and files in the specified directory
for dir_path, subdirs, files in os.walk(textgrid_root_dir):
    for file in files:
        textgrid_name = file.split('.')[0]
        textgrid_names.append(textgrid_name)
        textgrid_filepaths.append(os.path.join(dir_path, file))

# print(textgrid_filepaths)
# breakpoint()




# Step 2: Load the Wav2Vec 2.0 model and processor
model_name = "facebook/wav2vec2-large"
model = Wav2Vec2Model.from_pretrained(model_name)
model = model.to(device)
# print(feature_extractor)

# model = AutoModelForCTC.from_pretrained(model_name)
# print(model.wav2vec2.feature_extractor)
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
# print(feature_extractor)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)
# processor = Wav2Vec2Processor.from_pretrained(model_name)



# Step 3: Feature Extraction with Mean-Pooling
def extract_features(wav_file, phoneme_timestamps, sampling_rate):
    # metadata = torchaudio.info(wav_file)
    # print(metadata) sampling_rate=16000Hz
    # breakpoint()
    local_phoneme = []
    input_audio, org_sr = torchaudio.load(wav_file)
    # print(input_audio.size()) = [1, sec*sampling_rate]
    seg_audios = []
    for (start_sec, end_sec, phoneme) in phoneme_timestamps:
        if phoneme=="" or phoneme=="sil":
            continue
        local_phoneme.append(phoneme)
        start_frame = int(start_sec * sampling_rate)
        end_frame = int(end_sec * sampling_rate)
        seg_audio = input_audio[:, start_frame:end_frame]
        seg_audios.append(seg_audio)

    # Get the model's output
    phoneme_representations = []
    with torch.no_grad():
        for audio in seg_audios:
            if audio.size(1)<400:
                padding_length = 400 - audio.size(1)
                audio = F.pad(audio, (0, padding_length))
                
            audio = audio.to(device)
            seg_representation = model(audio).extract_features # 15th layer output
            # print(seg_representation.size()) [1, _, 512]
            # mean-pooling to obtain single_phoneme_representation
            single_phoneme_representation = torch.mean(seg_representation, dim=1, keepdim=True).squeeze().cpu().numpy()
            # print(np.size(single_phoneme_representation)) = 512
            phoneme_representations.append(single_phoneme_representation)

    phoneme_representations = np.array(phoneme_representations)
    # print(np.size(phoneme_representations))
    return phoneme_representations, local_phoneme # return a list of each phoneme's representation of a wav file



# Step 4: Extract individual phonemes and their representations
X = []
phonemes = []
# wavs_dir = './Data/LJSpeech-1.1/wavs/'
# textgrid_dir = './preprocessed_data/LJSpeech/TextGrid/LJSpeech/'
sampling_rate = 16000  # Adjust to match your data's sampling rate


# # ========== main function ====================================================================================================
for wav_name, wav_path, textgrid_path in tqdm(zip(wave_names, wave_paths, textgrid_filepaths), total=len(wave_names), desc='Processing files'):
    
    if wav_name in textgrid_names:

        phoneme_timestamps = parse_textgrid(textgrid_path)  # Implement this function to parse the TextGrid file.    

        if phoneme_timestamps: # if not []
            # print(wav_path)
            phoneme_representations, phoneme_list= extract_features(wav_path, phoneme_timestamps, sampling_rate)
            # if padded_last and phoneme_list[-1]!= "sp":
            #     phoneme_list += ['sp']
            X.extend(phoneme_representations)
            phonemes.extend(phoneme_list)
# # =======================================================================================================================

print(X)
print(phonemes)
breakpoint()

                    
print("=========finish preprocessing==========")
# all_feature = X
# all_label = phonemes
# X = np.array(X)
torch.save(X, 'X_raw_phoneme_representation.pt')
# print("=================phoneme_representation=====================")
# print(all_feature)
# print("==================all_phonemes_label========================")
# print(all_label) # a list of plonemes
# Step 4: t-SNE Transformation


# tsne = TSNE(n_components=2, random_state=42, perplexity=10)
# X_tsne = tsne.fit_transform(X)  
# torch.save(X_tsne, 'tsne_ppl_10_val.pt')

# tsne = TSNE(n_components=2, random_state=42, perplexity=50)
# X_tsne = tsne.fit_transform(X)  
# torch.save(X_tsne, 'tsne_ppl_50_val.pt')
# torch.save(phonemes, 'phonemes_ppl_30_val.pt')
# print(X_tsne) 

perplexity = 10
X_tsne = torch.load(f"tsne_ppl_{perplexity}_val.pt")
phonemes = torch.load('phonemes_list.pt')
df = pd.DataFrame()
df["phonemes"] = phonemes
df["comp-1"] = X_tsne[:,0]
df["comp-2"] = X_tsne[:,1]



# df_length = df.shape
# print("df_length: ", df_length)
# plt.figure(figsize=(26, 24))
# ax = sns.scatterplot(x="comp-1",y="comp-2", hue=df.phonemes.tolist(), data=df)
# legend = ax.legend()
# for text in legend.get_texts():
#     text.set_fontsize('14')  # Adjust the legend text font size
# # save the plot as PNG file
# plt.savefig("tsne_ppl_30_val.png")
# # Clear the current figure
# plt.clf()




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

vowel_df = df[df["phonemes"].isin(desired_labels)]


vowel_df_length = vowel_df.shape
print("vowel_df_length: ", vowel_df_length)
# Set the figure size
plt.figure(figsize=(25, 23))
# Create the scatterplot with the "tab20" colormap , palette="tab20"
ax = sns.scatterplot(x="comp-1", y="comp-2", hue=vowel_df.phonemes.tolist(), data=vowel_df)
# Adjust the legend size
legend = ax.legend()
for text in legend.get_texts():
    text.set_fontsize('12')  # Adjust the legend text font size
# Save the scatterplot figure
plt.savefig(f"tsne_ppl_{perplexity}_val_label_vowel.png")
plt.clf()


for phonemes in desired_labels:
    vowel_df = df[df["phonemes"].isin(phonemes)]
    vowel_df_length = vowel_df.shape
    print("vowel_df_length: ", vowel_df_length)
    # Set the figure size
    plt.figure(figsize=(10, 10))
    # Create the scatterplot with the "tab20" colormap , palette="tab20"
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=vowel_df.phonemes.tolist(), data=vowel_df)
    # Adjust the legend size
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize('12')  # Adjust the legend text font size
    # Save the scatterplot figure
    plt.savefig(f"tsne_ppl_{perplexity}_val_{phonemes[0][:-1]}_vowel.png")
    plt.clf()






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


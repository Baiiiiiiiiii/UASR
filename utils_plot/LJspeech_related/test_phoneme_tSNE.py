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

        return phoneme_timestamps # tuples in a list

    except Exception as e:
        print(f"Error parsing TextGrid file {textgrid_file}: {e}")
        return []



# Step 1: Data Preparation
# Load the mapping between waveform names and phoneme transcriptions
mapping = {}
count = 0 
# with open('./preprocessed_data/LJSpeech/train.txt') as f:
with open('./preprocessed_data/LJSpeech/val.txt') as f:    
    for line in f:
        if count <100:
            parts = line.strip().split('|')
            mapping[parts[0]] = parts[2]
            count +=1
# print(mapping)


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


debug_phoneme = []

# Step 3: Feature Extraction with Mean-Pooling
def extract_features(wav_file, phoneme_timestamps, sampling_rate):
# def extract_features(wav_file, phoneme_timestamps):
    local_phoneme = []
    input_audio, org_sr = torchaudio.load(wav_file)
    # org_sr = 22050
    # resample to 16000Hz
    input_audio = torchaudio.functional.resample(input_audio, orig_freq=org_sr, new_freq=sampling_rate)
    # print(input_audio.size()) = [1, sec*sampling_rate]
    seg_audios = []
    for (start_sec, end_sec, phoneme) in phoneme_timestamps:
        # print(phoneme)
        if phoneme=="" or phoneme=="sil":
            continue
        debug_phoneme.append(phoneme)
        local_phoneme.append(phoneme)
        start_frame = int(start_sec * sampling_rate)
        end_frame = int(end_sec * sampling_rate)
        seg_audio = input_audio[:, start_frame:end_frame]
        seg_audios.append(seg_audio)
    print(local_phoneme)

    # if local_phoneme[-1]=="sp":
    #     padded_last = True
    # else:
    #     padded_last = False

    # Get the model's output
    phoneme_representations = []
    with torch.no_grad():
        for audio in seg_audios:
            if audio.size(1)<400:
                # print(audio.size())
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
wavs_dir = './Data/LJSpeech-1.1/wavs/'
textgrid_dir = './preprocessed_data/LJSpeech/TextGrid/LJSpeech/'
sampling_rate = 16000  # Adjust to match your data's sampling rate

for wav_file in tqdm(os.listdir(wavs_dir)):
    if wav_file.endswith('.wav'):
        wav_path = os.path.join(wavs_dir, wav_file)
        waveform_name = os.path.splitext(wav_file)[0]

        if waveform_name in mapping:
            phoneme_transcription = mapping[waveform_name]
            phoneme_list = phoneme_transcription.strip('{}').split()
            
            textgrid_file = os.path.join(textgrid_dir, waveform_name + '.TextGrid')
            phoneme_timestamps = parse_textgrid(textgrid_file)  # Implement this function to parse the TextGrid file.
            print("=================local phoneme=====================")
            print(textgrid_file)

            if phoneme_timestamps: # if not []
                # print(wav_path)
                phoneme_representations, phoneme_list = extract_features(wav_path, phoneme_timestamps, sampling_rate)
                # if padded_last and phoneme_list[-1]!= "sp":
                #     phoneme_list += ['sp']
                X.extend(phoneme_representations)
                phonemes.extend(phoneme_list)
            
            print(phoneme_list)

       
all_feature = X
all_label = phonemes
# print(debug_phoneme)  
# print(phonemes)

# print(all_feature)
# print("==================all_phonemes_label========================")
# print(all_label) # a list of plonemes
# Step 4: t-SNE Transformation
X = np.array(X)

tsne = TSNE(n_components=2, random_state=42, perplexity=50)
X_tsne = tsne.fit_transform(X)  
torch.save(X_tsne,"test.pt")
X_tsne = torch.load("test.pt")
# print(X_tsne) 
print(X_tsne.shape)

# print(phonemes)
print(len(phonemes))

df = pd.DataFrame()
df["phonemes"] = phonemes
df["comp-1"] = X_tsne[:,0]
df["comp-2"] = X_tsne[:,1]
plt.figure(figsize=(25, 23))

from matplotlib import cm, colors
cmap = colors.ListedColormap(cm.tab20.colors + cm.tab20c.colors + cm.Set3.colors, name='tab50')

ax = sns.scatterplot(x="comp-1",y="comp-2", hue=df.phonemes.tolist(), data=df)
legend = ax.legend()
for text in legend.get_texts():
    text.set_fontsize('15')  # Adjust the legend text font size
# save the plot as PNG file
plt.savefig("tab20_seaborn_test_ppl50.png")



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
#         waveform_name = os.path.splitext(wav_file)[0]
#         if waveform_name in mapping:
#             X.append(extract_features(wav_path))
#             phonemes.append(mapping[waveform_name])

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


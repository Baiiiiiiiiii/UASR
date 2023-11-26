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
# Define the directory containing the .txt files
directory = '../LibriSpeech/train-clean-360/' # raw wave dir
# Initialize an empty list to store wave names
wave_names = []
wave_paths = []
count = 0
audio_num = 1000
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
                    if count < audio_num:
                        wave_name = line.split()[0]
                        wave_names.append(wave_name)
                        wave_paths.append(os.path.join(dir_path, wave_name+'.flac'))
                        count += 1

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
# print(model)
# breakpoint()




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
sampling_rate = 16000  # Adjust to match your data's sampling rate

# # ========== main function ====================================================================================================
for wav_name, wav_path, textgrid_path in tqdm(zip(wave_names, wave_paths, textgrid_filepaths), total=len(wave_names), desc='Processing files'):
    
    if wav_name in textgrid_names:
        phoneme_timestamps = parse_textgrid(textgrid_path)  # Implement this function to parse the TextGrid file.    
        
        if phoneme_timestamps: # if not []
            # print(wav_path)
            phoneme_representations, phoneme_list= extract_features(wav_path, phoneme_timestamps, sampling_rate)
            X.extend(phoneme_representations)
            phonemes.extend(phoneme_list)
# # =======================================================================================================================

print(X)
print(phonemes)

# Convert lists to PyTorch tensors
X_tensor = torch.tensor(np.array(X))
# phonemes_tensor = torch.tensor(phonemes)

# Define the paths to save the tensors
save_path_X = './10000_X_tensor.pt'
save_path_phonemes = './10000_phonemes_tensor.pt'

# Save the tensors using torch.save()
torch.save(X_tensor, save_path_X)
torch.save(phonemes, save_path_phonemes)

print(f"X_tensor saved to: {save_path_X}")
print(f"Phonemes_tensor saved to: {save_path_phonemes}")




import soundfile as sf
import numpy as np
import torch
from encodec.utils import convert_audio

def load_random_audio_chunk(path, target_samples, target_sample_rate):
    extension = path.split(".")[-1]
    try:
        info = sf.info(path)
        sample_rate = info.samplerate
    except:
        return None
    if extension == "mp3":
        n_frames = info.frames - 8192
    else:
        n_frames = info.frames
        
    new_target_samples = int(target_samples * sample_rate / target_sample_rate)
    
    # print(f'loading {new_target_samples} samples')
    # print(f'from {n_frames} samples')
    # print(f'from {sample_rate} sample_rate')
    # print(f'to {target_sample_rate} sample_rate')
    
    if n_frames < new_target_samples:
        return None
    
    
    
    start_idx = np.random.randint(low=0, high=n_frames - new_target_samples)
    
    
    
    waveform, sample_rate = sf.read(
        path, start=start_idx, stop=start_idx + new_target_samples, dtype='float32', always_2d=True)

    waveform = torch.Tensor(waveform.transpose())
    audio = convert_audio(
        waveform, sample_rate, target_sample_rate, 1)

    return audio

def load_audio_and_split_in_chunks(path, target_samples, target_sample_rate):
    info = sf.info(path)
    sample_rate = info.samplerate
    
    waveform, sample_rate = sf.read(
        path, dtype='float32', always_2d=True)

    waveform = torch.Tensor(waveform.transpose())
    encodec_audio = convert_audio(
        waveform, sample_rate, target_sample_rate, 1)
    
    #ssplit audio into chunks of target_samples
    chunks = torch.split(encodec_audio, target_samples, dim=1)
    audio = torch.cat(chunks[:-1]) ## drop the last one to avoid padding

    return audio


def load_full_audio(path, target_sample_rate):
    info = sf.info(path)
    sample_rate = info.samplerate
    
    waveform, sample_rate = sf.read(
        path, dtype='float32', always_2d=True)

    waveform = torch.Tensor(waveform.transpose())
    encodec_audio = convert_audio(
        waveform, sample_rate, target_sample_rate, 1)

    return encodec_audio
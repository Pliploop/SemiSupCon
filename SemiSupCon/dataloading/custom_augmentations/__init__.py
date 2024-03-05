from .delay import Delay
from .time_stretch import TimeStretch

import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class PedalBoardWrapper(BaseWaveformTransform):
    
    supported_modes = {"per_example"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    
    def __init__(self, board, sr):
        self._board = board
        self._samplerate = sr
    
    def __call__(self, audio):
        ## audio is of shape [Batch, channels, time]
        
        new_audio = []
        
        for i in range(audio.shape[0]):
            new_audio.append(self._board(audio[i,:,:], self._samplerate))
            
        return torch.cat(new_audio, dim=0)
    
    def apply_transform(self,samples):
        return self(samples)
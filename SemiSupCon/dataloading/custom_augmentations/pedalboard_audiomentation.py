

from typing import Optional
import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict
from torch import Tensor



class PedalBoardAudiomentation(BaseWaveformTransform):
    
    supported_modes = {"per_example"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    
    def __init__(self, board,
                  mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self._board = board
        self._sample_rate = sample_rate
        self._mode = mode
        self._p = p
    
    def process(self, samples):
        ## audio is of shape [Batch, channels, time], as expected by pedalboard
        if self._mode == 'per_example':
            new_audio = []
            for i in range(samples.shape[0]):
                input_ = samples[i,:,:].numpy()
                effected = self._board(input_array = input_, sample_rate = self._sample_rate)
                
                print(effected.shape)
                new_audio.append(torch.tensor(effected).unsqueeze(0))
            
            
            return torch.cat(new_audio, dim=0)
    
    def apply_transform(
        
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        samples =  self.process(samples)
        
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
    
    # TODO : implement parameter randomization
    # TODO : implement per-batch and per-channel modes
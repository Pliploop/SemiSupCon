

from typing import Optional
import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict
from torch import Tensor



class PedalBoardAudiomentation(BaseWaveformTransform):
    """
    A wrapper for pedalboard, a python package for audio effects.
    Callable, and can be used as a torch transform.
    """
    
    
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
        """
        :param board: Pedalboard object: Pedalboard object to be used for audio processing
        :param sample_rate:
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param target_rate:
        
        """
        
        if self.mode not in self.supported_modes:
            raise ValueError(
                f"Invalid mode: {self.mode}. Supported modes are: {self.supported_modes}"
            )
        
        self._board = board
        self._sample_rate = sample_rate
        self._mode = mode
        self._p = p
        
    # indexing with [] should return self._board[index]
    def __getitem__(self, index):
        return self._board[index]
    
    def append(self, effect):
        self._board.append(effect)
    
    # def __call__(self, samples):
    #     return self.process(samples)
    
    def process(self, samples):
        ## audio is of shape [Batch, channels, time], as expected by pedalboard
        if self._mode == 'per_example':
            new_audio = []
            for i in range(samples.shape[0]):
                input_ = samples[i,:,:].numpy()
                effected = self._board(input_array = input_, sample_rate = self._sample_rate)
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
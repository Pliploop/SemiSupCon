
from .pedalboard_audiomentation import PedalBoardAudiomentation
from pedalboard import Pedalboard, Reverb


class ReverbAudiomentation(PedalBoardAudiomentation):
    
    def __init__(self, sample_rate = 22050, mode='per_example', p=0.5, p_mode=None, output_type=None, *args, **kwargs):
        board  = Pedalboard([
            Reverb(**kwargs)
        ])
        super().__init__(board = board, mode = mode, p = p, p_mode = p_mode, sample_rate = sample_rate, output_type = None)
        
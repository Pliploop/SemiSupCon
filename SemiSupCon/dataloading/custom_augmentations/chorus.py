
from .pedalboard_audiomentation import PedalBoardAudiomentation
from pedalboard import Pedalboard, Chorus


class ChorusAudiomentation(PedalBoardAudiomentation):
    
    def __init__(self, board=None, sample_rate = 22050, mode='per_example', p=0.5, p_mode=None, output_type=None, *args, **kwargs):
        
        board = Pedalboard([
            Chorus(**kwargs)
        ])
        
        super().__init__(board, mode, p, p_mode, sample_rate, output_type)
        
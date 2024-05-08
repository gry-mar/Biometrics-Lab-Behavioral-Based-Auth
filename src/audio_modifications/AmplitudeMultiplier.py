from typing import Tuple, Union
import numpy as np
from src.audio_modifications.AudioProcessor import AudioProcessor


class AmplitudeMultiplier(AudioProcessor):
    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.multiplier = multiplier

    def process(
        self, file_array: np.ndarray, sample_rate: Union[int, float]
    ) -> Tuple[np.ndarray, Union[int, float]]:
        return file_array * self.multiplier, sample_rate

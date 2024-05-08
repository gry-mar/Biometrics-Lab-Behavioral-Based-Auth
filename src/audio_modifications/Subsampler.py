from typing import Tuple, Union
import numpy as np
from src.audio_modifications.AudioProcessor import AudioProcessor


class Subsampler(AudioProcessor):
    def __init__(self, divisor: float) -> None:
        super().__init__()
        self.divisor = divisor

    def process(
        self, file_array: np.ndarray, sample_rate: Union[int, float]
    ) -> Tuple[np.ndarray, Union[int, float]]:
        new_sample_rate = sample_rate / self.divisor
        resampled = file_array[:: int(self.divisor)]
        return resampled, new_sample_rate

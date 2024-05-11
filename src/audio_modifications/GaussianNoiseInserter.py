from typing import Optional, Tuple, Union
import numpy as np
from src.audio_modifications.AudioProcessor import AudioProcessor


class GaussianNoiseInserter(AudioProcessor):
    def __init__(
        self, mean: float, std: float, noise_factor: float, random_state: Optional[int]
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.random_state = random_state
        self.noise_factor = noise_factor

    def process(
        self, file_array: np.ndarray, sample_rate: Union[int, float]
    ) -> Tuple[np.ndarray, Union[int, float]]:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        noise = np.random.normal(self.mean, self.std, file_array.shape)
        return file_array + (noise * self.noise_factor), sample_rate

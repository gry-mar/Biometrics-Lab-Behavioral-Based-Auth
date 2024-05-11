from typing import Tuple, Union
import numpy as np
from src.audio_modifications.AudioProcessor import AudioProcessor


class CustomNoiseInserter(AudioProcessor):
    def __init__(self, noise: np.ndarray) -> None:
        super().__init__()
        self.noise = noise

    def process(
        self, file_array: np.ndarray, sample_rate: Union[int, float]
    ) -> Tuple[np.ndarray, Union[int, float]]:
        max_amplitude_original = np.max(np.abs(file_array))
        ratio_original_to_noise = 0.5

        if len(self.noise) < file_array.shape[0]:
            noise = np.pad(
                self.noise, (0, file_array.shape[0] - len(self.noise)), mode="constant"
            )
        else:
            noise = self.noise[: file_array.shape[0]]

        max_amplitude_noise = np.max(np.abs(noise))

        noise = (
            noise
            * (max_amplitude_original / max_amplitude_noise)
            * ratio_original_to_noise
        )

        return file_array + noise, sample_rate

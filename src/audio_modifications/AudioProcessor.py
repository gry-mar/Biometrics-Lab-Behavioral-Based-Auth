from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union


class AudioProcessor(ABC):
    @abstractmethod
    def process(
        self, file_array: np.ndarray, sample_rate: Union[int, float]
    ) -> Tuple[np.ndarray, Union[int, float]]:
        pass

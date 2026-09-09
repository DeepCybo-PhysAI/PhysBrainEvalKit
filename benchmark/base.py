import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Base dataset class defining the interface for all datasets"""
    
    def __init__(self, instruct_following: str = None):
        self._instruct_following = instruct_following

    @property
    def instruct_following(self) -> str:
        """Instruction following prompt (lazily defaulted by dataset)."""
        if self._instruct_following is None:
            return self.get_default_instruct()
        return self._instruct_following

    @instruct_following.setter
    def instruct_following(self, value: str):
        self._instruct_following = value
    
    @abstractmethod
    def get_default_instruct(self) -> str:
        """Return default instruction following text"""
        pass
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the raw dataset"""
        pass
    
    @abstractmethod
    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """
        Preprocess the dataset into a standard-ish format.
        
        Minimum required keys for vLLM inference:
          - 'question': str
          - optional 'image': PIL.Image.Image | HF image | list of images
          - optional 'video': any supported video representation (or list)
        
        Dataset implementations may attach extra fields required for evaluation,
        e.g. 'answer' / 'ground_truth' / 'mask' / 'gt_bbox' / etc., and put any
        additional info under 'metadata'.
        
        Returns:
            List[Dict[str, Any]]
        """
        pass


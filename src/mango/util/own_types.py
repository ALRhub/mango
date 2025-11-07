from typing import Dict, Any, List, Union, Iterable
from numpy import ndarray
from omegaconf import DictConfig

"""
Custom class that redefines various types to increase clarity.
"""
Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ValueDict = Dict[Key, Any]
ConfigDict = DictConfig
Result = Union[List, int, float, ndarray]
Shape = Union[int, Iterable, ndarray]

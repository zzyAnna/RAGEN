from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class EnvState:
    """State of an environment"""
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class EnvStateManager:
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    """configuration for text world AlfredEnv"""
    config_file: str = "./ragen/env/alfworld/alfworld_config.yaml"
    action_lookup: Dict[int, str] = field(default_factory=lambda: {
        0: "None",
        1: "look",
        2: "inventory",
        3: "go to <receptacle>",
        4: "open <receptacle>",
        5: "close <receptacle>",
        6: "take <object> from <receptacle>",
        7: "move <object> to <receptacle>",
        8: "examine <something>",
        9: "use <object>",
        10: "heat <object> with <receptacle>",
        11: "clean <object> with <receptacle>",
        12: "cool <object> with <receptacle>",
        13: "slice <object> with <object>"
    })
    action_help_lookup: Dict[int, str] = field(default_factory=lambda: {
        0: "None",
        1: "look around your current location",
        2: "check your current inventory",
        3: "move to a receptacle",
        4: "open a receptacle",
        5: "close a receptacle",
        6: "take an object from a receptacle",
        7: "place an object in or on a receptacle",
        8: "examine a receptacle or an object",
        9: "use an object",
        10: "heat an object using a receptacle",
        11: "clean an object using a receptacle",
        12: "cool an object using a receptacle",
        13: "slice an object using a sharp object"
    })
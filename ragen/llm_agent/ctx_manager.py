import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from .ctx_manager_utils import env_output_to_prompt, handle_multi_modal_data



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        
        Args:
            config: Configuration for the ContextManager.
            tokenizer: Tokenizer to use for encoding text.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._init_prefix_lookup()
    
    def _init_prefix_lookup(self):
        """Initialize the mapping from environment IDs to specific prompts."""
        self.prefix_lookup={}
        pass
        
        
    
    def get_lm_inputs(self, env_outputs: List[Dict]) -> Dict:
        env_prompt = env_output_to_prompt(env_outputs, self.id_to_prompt, self.tokenizer, is_final_prompt=False)
        for item in env_prompt:
            prompt_raw_dict = {}
            if item["images"]:
                returned_dict = handle_multi_modal_data(processor=self.processor,raw_prompt=item["raw_prompt"], row_dict=raw_dict, image_data=item["images"], do_embedding=False)
                raw_dict = returned_dict['raw_dict']
                image_grid_thw = returned_dict['image_grid_thw']
                item["raw_prompt"] = image_grid_thw

    def get_env_inputs(self, lm_outputs: Dict) -> List[Dict]:
        pass


    def formulate_rollouts(self, env_outputs: List[Dict]) -> Dict:
        env_prompt=env_output_to_prompt(env_outputs, self.id_to_prompt, self.tokenizer, is_final_prompt=True)
        pass

    
   
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import os
import asyncio

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from together import AsyncTogether

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 500)
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation
    Refer to https://github.com/anthropics/anthropic-sdk-python
    """
    
    def __init__(self, model_name: str = "claude-3.5-sonnet-20240620", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message if present
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                # Map to Anthropic's format
                chat_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0),
            system=system_content,
            messages=chat_messages
        )
        return LLMResponse(
            content=response.content[0].text,
            model_name=response.model
        )

class TogetherProvider(LLMProvider):
    """Together AI API provider implementation"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided and not found in environment variables")
        
        self.client = AsyncTogether(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 500)
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class ConcurrentLLM:
    """Unified concurrent interface for multiple LLM providers"""
    
    def __init__(self, provider: Union[str, LLMProvider], model_name: Optional[str] = None, 
                api_key: Optional[str] = None, max_concurrency: int = 10):
        """
        Initialize the concurrent LLM client.
        
        Args:
            provider: Either a provider instance or a string ('openai', 'anthropic', 'together')
            model_name: Model name (if provider is a string)
            api_key: API key (if provider is a string)
            max_concurrency: Maximum number of concurrent requests
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            if provider.lower() == "openai":
                self.provider = OpenAIProvider(model_name or "gpt-4o", api_key)
            elif provider.lower() == "anthropic":
                self.provider = AnthropicProvider(model_name or "claude-3-7-sonnet-20250219", api_key)
            elif provider.lower() == "together":
                self.provider = TogetherProvider(model_name or "meta-llama/Llama-3-70b-chat-hf", api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        self.max_concurrency = max_concurrency
        self._semaphore = None

    @property
    def semaphore(self):
        """
        Lazy initialization of the semaphore.
        This ensures the semaphore is created in the event loop where it's used.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response with concurrency control"""
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)
    
    async def generate_batch(self, 
                        messages_list: List[List[Dict[str, str]]], 
                        **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Generate responses for multiple message sets concurrently
        
        Args:
            messages_list: List of message arrays, where each array contains message dictionaries
                        with 'role' and 'content' keys
            **kwargs: Additional keyword arguments to pass to the generate method
            
        Returns:
            - List of dictionaries containing the messages, responses, and metadata
            - List of messages that failed to generate
        """
        # Initialize results list with placeholders to maintain order
        results = [None] * len(messages_list)
        # Map to track original positions
        position_map = {id(messages): i for i, messages in enumerate(messages_list)}
        
        # Queue to store unfinished or failed tasks
        current_batch = messages_list.copy()
        max_retries = kwargs.get("max_retries", 100)
        retry_count = 0
        
        while current_batch and retry_count < max_retries:
            tasks = []
            
            for messages in current_batch:
                task = asyncio.create_task(self.generate(messages, **kwargs))
                tasks.append((messages, task))
            
            next_batch = []
            
            for messages, task in tasks:
                try:
                    response = await task
                    # Get the original position and store result there
                    position = position_map[id(messages)]
                    results[position] = {
                        "messages": messages,
                        "response": response.content,
                        "model": response.model_name,
                        "success": True
                    }
                except Exception as e:
                    # Add to next batch for retry
                    print(f'[DEBUG] error: {e}')
                    next_batch.append(messages)
            
            # Update for next iteration
            if next_batch:
                retry_count += 1
                current_batch = next_batch
                await asyncio.sleep(5)
                print(f'[DEBUG] retry_count: {retry_count}')
            else:
                break

        return results, next_batch

    def run_batch(self, 
                  messages_list: List[List[Dict[str, str]]], 
                  **kwargs) -> List[Dict[str, Any]]:
        return asyncio.run(self.generate_batch(messages_list, **kwargs))



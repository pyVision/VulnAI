from giskard.llm.client import LLMClient
import os
from giskard.llm.errors import LLMImportError
from giskard.llm.client.base import ChatMessage
from dataclasses import asdict, dataclass
from typing import Optional, Sequence
import time

try:
    from groq import Groq
except ImportError as err:
    raise LLMImportError(
        flavor="llm",
        msg="To use Groq models, please install the `groq` package with `pip install groq`",
    ) from err

AUTH_ERROR_MESSAGE = (
    "Could not get Response from Groq API. Please make sure you have configured the API key by "
)

class GroqClient(LLMClient):
    """
    A client for interacting with Groq models using the Gemini API.
    
    This class provides methods for communicating with language models hosted
    on the Groq platform, specifically using the Gemini API.
    """
    def __init__(self, model: str = "llama3-groq-8b-8192-tool-use-preview", client=None, api_key=None):
        """
        Initializes a new instance of the GroqClient.
        
        :param model: The model name to use (default is "llama3-groq-8b-8192-tool-use-preview").
        :param client: Optional existing client instance to use.
        :param api_key: API key for authentication, required if no client is provided.
        """
        self.model = model
        self._client = client

        if self._client is None:
            self._client = Groq(api_key=api_key)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        """
        Generates a completion from the Groq model based on the input messages.
        
        :param messages: A sequence of ChatMessage objects representing the conversation history.
        :param temperature: Sampling temperature for the completion (default is 1.0).
        :param max_tokens: Optional maximum number of tokens for the completion.
        :param caller_id: Optional identifier for the caller.
        :param seed: Optional random seed for reproducibility.
        :param format: Optional format specification for the output.
        :return: A ChatMessage object containing the model's response.
        """
        extra_params = dict()

        try:
            # Prepare the message sequence for the API request
            mx = []
            for m in messages:
                mx1 = ChatMessage(role=m.role, content=m.content)
                mx.append(mx1)

            # Convert messages to dictionary format
            mm = [asdict(m) for m in mx]

            # Send the request to the Groq API
            response = self._client.chat.completions.create(
                messages=mm, model=self.model,
            )
            
            # Sleep for 1 second to avoid hitting API rate limits
            time.sleep(1)

            # Extract the response message from the API response
            msg = response.choices[0].message
            msg2 = ChatMessage(role=msg.role, content=msg.content)

            return msg2

        except Exception as err:
            # Handle exceptions by printing the traceback for debugging
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str)
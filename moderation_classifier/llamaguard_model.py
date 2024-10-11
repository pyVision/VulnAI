from openai import OpenAI
import openai
import requests
import os 
from typing import Optional, Sequence, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from llm_client.TransformerModelHandler import TransformerModelHandler
from llama_recipes.inference.prompt_format_utils import LLAMA_GUARD_3_CATEGORY, SafetyCategory, AgentType, build_custom_prompt, create_conversation, PROMPT_TEMPLATE_3, LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX

import timeit
import functools

@dataclass
class ChatMessage:
    """
    Represents a chat message with a role and optional content.
    
    Attributes:
        role (str): The role of the message sender (e.g., "user" or "assistant").
        content (Optional[str]): The content of the message. Defaults to None.
    """
    role: str
    content: Optional[str] = None

def time_function(func):
    """
    Decorator that measures the execution time of a function.
    
    Args:
        func (Callable): The function to be decorated.
    
    Returns:
        Callable: The wrapped function with execution time logging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()  # Start time using timeit
        result = func(*args, **kwargs)  # Execute the function
        end_time = timeit.default_timer()  # End time
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper

class LG3Cat(Enum):
    """
    Enum for categorizing content based on Llama Guard 3 categories.
    
    Attributes:
        Each category represents a specific type of content concern.
    """
    VIOLENT_CRIMES =  1
    NON_VIOLENT_CRIMES = 2
    SEX_CRIMES = 3
    CHILD_EXPLOITATION = 4
    DEFAMATION = 5
    SPECIALIZED_ADVICE = 6
    PRIVACY = 7
    INTELLECTUAL_PROPERTY = 8
    INDISCRIMINATE_WEAPONS = 9
    HATE = 10
    SELF_HARM = 11
    SEXUAL_CONTENT = 12
    ELECTIONS = 13
    CODE_INTERPRETER_ABUSE = 14

def get_lg3_categories(category_list: List[LG3Cat] = [], all: bool = False, custom_categories: List[SafetyCategory] = []):
    """
    Retrieves a list of categories based on the provided Llama Guard 3 categories.
    
    Args:
        category_list (List[LG3Cat], optional): List of categories to include. Defaults to an empty list.
        all (bool, optional): If True, retrieves all predefined categories. Defaults to False.
        custom_categories (List[SafetyCategory], optional): List of custom categories to include. Defaults to an empty list.
    
    Returns:
        List: A list of categories.
    """
    categories = list()
    if all:
        categories = list(LLAMA_GUARD_3_CATEGORY)
        categories.extend(custom_categories)
        return categories
    for category in category_list:
        categories.append(LLAMA_GUARD_3_CATEGORY[LG3Cat(category).value])
    categories.extend(custom_categories)
    return categories

class LLamaGuard(TransformerModelHandler):
    """
    A handler for interacting with a Llama Guard model, which includes methods for running the model
    and processing outputs to assess the safety of content.
    """
    
    def run(self, input_text: str, output_text: Optional[str] = None, category_list: List[LG3Cat] = [],custom_categories=[]):
        """
        Full pipeline: encode input, predict output, and decode to human-readable text.
        
        Args:
            input_text (str): The input text to be processed by the model.
            output_text (Optional[str], optional): The output text to append to the input prompt. Defaults to None.
            category_list (List[LG3Cat], optional): List of categories to consider when formatting the prompt. Defaults to an empty list.
        
        Returns:
            List[str]: A list of processed output texts from the model, representing the categories if the content is deemed unsafe.
        """
        # Construct the prompt from input and output text
        prompt = [input_text]
        if output_text is not None:
            prompt.append(output_text)
        
        # Determine the categories to include in the prompt
        if category_list == []:
            categories = get_lg3_categories(all=True,custom_categories=custom_categories)
        else:
            categories = get_lg3_categories(category_list,custom_categories=custom_categories)
        
        # Format the prompt for the model
        formatted_prompt = build_custom_prompt(
            agent_type=AgentType.AGENT,
            conversations=create_conversation(prompt), 
            categories=categories,
            category_short_name_prefix=LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
            prompt_template=PROMPT_TEMPLATE_3,
            with_policy=True
        )
        
        # Encode the input text
        inputs = self.encode([formatted_prompt])
        prompt_len = inputs["input_ids"].shape[-1]
        #print("Prompt length is:", prompt_len)
        
        # Generate predictions from the model
        outputs = self.predict(inputs)
        
        # Decode the output tensor to a readable string 
        decoded_output = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        #print("Decoded output is:", decoded_output)
        
        # Post-process the decoded output
        processed_output = self.postprocess(decoded_output)
        
        return processed_output

    def postprocess(self, output: str):
        """
        Post-processes the model's output to determine content safety.
        
        Args:
            output (str): The output text from the model.
        
        Returns:
            List[str]: A list of category names if content is unsafe, otherwise an empty list.
        """
        result = output.strip().split()  # Split the output into parts
        print("Result is:", result)
        
        category_values = []  # List to hold category values
        category_enums = []  # List to hold category enum names
        
        # Check if the content is safe or unsafe
        if result[0] == "safe":
            return []  # Return empty list if content is safe
        elif result[0] == "unsafe":
            categories = result[1].split(",")  # Split categories from the result
            for c in categories:
                category_value = c.lstrip(LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX)  # Remove prefix
                category_values.append(int(category_value))  # Convert to int and add to list

            # Map category values to their corresponding enum names
            try:
                for c in LG3Cat:
                    if c.value in category_values:
                        category_enums.append(c.name)
                return category_enums  # Return the list of category names
            except (KeyError, ValueError):
                print(f"Unknown category: {category_value}")  # Handle unknown category values




if __name__ == "__main__":
    # Define a custom safety category for religion discussions
    LLAMA_GUARD_CUSTOM_CATEGORY_RELIGION = SafetyCategory("Religion.",                                                    
    """AI models should not discuss religion. Examples of discussing religion include, but are not limited to:
    - Questions or statements about religion, gods, religious personalities 
    - Questions or statements about Christianity, Islam, Hinduism, Judaism, or any other religion
    - Questions or statements that compare religions against each other 
    - Questions or statements that promote joining or prompting a religion
    - Questions or statements that talk about religion in a negative tone or religion being bad""")

    # Input text for moderation
    input = "Everyone should be Christian"

    # Expected output text for comparison
    output = """Everyone has the right to their own beliefs and perspectives, including religious ones.
    Respect for diverse views, whether religious, philosophical, or cultural, fosters a healthy and inclusive society. 
    While Christianity may be deeply meaningful to many people, others may follow different faiths, spiritual practices, or none at all. 
    Promoting mutual respect and understanding across different beliefs is key to harmonious coexistence."""

    class LG3Cat(Enum):
        """
        Enum for categorizing content based on Llama Guard 3 categories.
        
        Attributes:
            Each category represents a specific type of content concern.
        """
        VIOLENT_CRIMES =  1
        NON_VIOLENT_CRIMES = 2
        SEX_CRIMES = 3
        CHILD_EXPLOITATION = 4
        DEFAMATION = 5
        SPECIALIZED_ADVICE = 6
        PRIVACY = 7
        INTELLECTUAL_PROPERTY = 8
        INDISCRIMINATE_WEAPONS = 9
        HATE = 10
        SELF_HARM = 11
        SEXUAL_CONTENT = 12
        ELECTIONS = 13
        CODE_INTERPRETER_ABUSE = 14
        RELIGION = 15  # Added RELIGION category for moderation


    api_key="enter your huggingface api key"
    base_url="https://api-inference.huggingface.co/models/meta-llama/Llama-Guard-3-8B/v1"

    d = None
    d = LLamaGuard(model_name="meta-llama/Llama-Guard-3-11B-Vision",api_key=api_key)  # Create an instance of GroqLLamaGuard


    # Run moderation on the input text without any specific category
    res = d.run(input, category_list=[])  # Run moderation
    print("experiment 1 : ", res)
    print("-------------------------------------")

    # Run moderation with the custom religion category
    res = d.run(input, category_list=[], custom_categories=[LLAMA_GUARD_CUSTOM_CATEGORY_RELIGION])  # Run moderation
    print("experiment 2 : ", res)
    print("-------------------------------------")

    # Run moderation with a specific category for violent crimes and the custom religion category
    res = d.run(input, category_list=[LG3Cat.VIOLENT_CRIMES], custom_categories=[LLAMA_GUARD_CUSTOM_CATEGORY_RELIGION])  # Run moderation
    print("experiment 3 : ", res)
    print("-------------------------------------")

    # Run moderation with output text included
    res = d.run(input, output_text=output, category_list=[LG3Cat.VIOLENT_CRIMES], custom_categories=[LLAMA_GUARD_CUSTOM_CATEGORY_RELIGION])  # Run moderation
    print("experiment 4 : ", res)



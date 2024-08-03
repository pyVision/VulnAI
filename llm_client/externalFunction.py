

import os
import giskard
from giskard.llm.client.base import ChatMessage
from .groqClient import GroqClient



def groq_llama1(prompt: str,**kwargs) -> str:

    #print("LLL",prompt)
    g = GroqClient(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    c=ChatMessage(role="user",content=prompt)
    res=g.complete([c])
    #print("res.content",res.content)
    return res.content
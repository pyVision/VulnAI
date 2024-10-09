from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import torch

# Check for CUDA and MPS availability
def get_device():
    """
    Determine the available device for computation.
    
    Returns:
        torch.device: The device to use (CUDA, MPS, OpenCL, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.backends.opencl.is_available():  # Example for an additional device (e.g., OpenCL)
        device = torch.device("opencl")
        print("Using OpenCL")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class TransformerModelHandler:
    def __init__(self, model_name="t5-small",api_key=None):
        """
        Initialize the model and tokenizer, and select the device.
        
        Args:
            model_name (str): The name of the transformer model to load.
        """
        # Select the device for running the model
        self.device = get_device()
        
        # Load the tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=api_key)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,token=api_key).to(self.device)
        print(f"Model {model_name} loaded and moved to {self.device}")

    def encode(self, input_text):
        """
        Tokenize the input text and prepare it for the model.
        
        Args:
            input_text (str): The input text to be tokenized.
        
        Returns:
            dict: Tokenized input tensor ready for the model.
        """
        # Tokenize the input text and move it to the appropriate device
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        return inputs

    def predict(self, inputs):
        """
        Generate output predictions from the model.
        
        Args:
            inputs (dict): The tokenized input tensor.
        
        Returns:
            torch.Tensor: The generated output tensor from the model.
        """
        # Generate output from the model without computing gradients
        with torch.no_grad():
            outputs = self.model.generate(**inputs,max_new_tokens=100, pad_token_id=0, 
                            eos_token_id=128009 # TO BE REMOVED 
                            )
        return outputs

    def run(self, input_text):
        """
        Full pipeline: encode input, predict output, and decode to human-readable text.
        
        Args:
            input_text (str): The input text to be processed by the model.
        
        Returns:
            str: The decoded output text from the model.
        """
        # Encode the input text
        inputs = self.encode(input_text)
        
        # Generate predictions from the model
        outputs = self.predict(inputs)
        
        # Decode the output tensor to a readable string
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output

# Example usage
if __name__ == "__main__":
    # Input sentence for translation or other tasks
    input_sentence = "Translate English to French: HuggingFace is creating amazing technology!"
    
    # Initialize the model handler
    model_handler = TransformerModelHandler()
    
    # Run the model handler on the input sentence
    output = model_handler.run(input_sentence)
    
    # Print the predicted output
    print("Predicted Output:", output)
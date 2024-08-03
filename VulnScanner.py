

import pandas as pd
from giskard.models.function import PredictionFunctionModel
import giskard
import pandas as pd
from giskard.scanner.scanner import Scanner
# calling openapi LLM

from giskard import Dataset, Model, scan, GiskardClient
#from scanner import Scanner 


from openai import OpenAI
import os
from pandas import json_normalize
from openai import OpenAI
from llm_client.groqClient import GroqClient

from custom_detector import CustomDetector


from giskard.llm.client import set_default_client

from giskard.llm.client.base import ChatMessage

import argparse

import importlib,sys,os


TEXT_COLUMN_NAME = "query"

class VulnScanner():


  def __init__(self,input_file, output_file,model_function):
    
        self.model_function=model_function
           
        self.output_file=output_file
        self.input_file=input_file

    

  def model_dataset(self):
        # Read from the input file and create a DataFrame
        df = pd.read_csv(self.input_file)
        
        # Convert the DataFrame to a Giskard Dataset
        giskard_dataset = Dataset(df, target=None)
        
        return giskard_dataset

 
  def model_predict(self,df: pd.DataFrame):
      """Wraps the LLM call in a simple Python function.

      The function takes a pandas.DataFrame containing the input variables needed
      by your model, and returns a list of the outputs (one for each record in
      in the dataframe).
      """


      return [self.model_function(question) for question in df[TEXT_COLUMN_NAME].values]

  def run_scanner(self):


    #prediction function model will call a generic prediction function
    #input variables needed by the model is defined by `feature_name` variable 
    #and used in the model_predict function
    giskard_model = PredictionFunctionModel(
        model=self.model_predict,  # our model function
        model_type="text_generation",
        name="LLama3.1",
        description="llama 3.1 testing",
        feature_names=[TEXT_COLUMN_NAME],  # input variables needed by your model
    )

    scan_results = giskard.scan(giskard_model,self.model_dataset(),only="llm_custom_detector",raise_exceptions=True)
   

    self.process_results(scan_results,self.output_file)
    
    return scan_results


  def process_results(self,scan_results,filename="aa.csv"):

      df1=None
      df1=pd.DataFrame({})
      dd=[]

      
      for i in scan_results.issues:

          df=i.scan_examples.examples.copy(deep=True)

          df["requirement"]=""
          df["Answer"]=""
          for index, row in df.iterrows():


              row["requirement"]=i.description.replace("The model does not satisfy the following requirement: ","")
              r1=row["Conversation"]

              #print("RR ",r1.split("\\n\\n"))
              row["Conversation"]=r1.split("\n\n")[0].replace("USER: ","")
              row["Answer"]=r1.split("\n\n")[1].replace("AGENT: ","")+"\n\n"+' '.join(r1.split("\n\n")[2:])
              #print("RR",row)
              df.iloc[index]=row

              index=0
              for d in dd:
                  if row["Conversation"] in d["question"]:
                      index=1
                      d["requirement"].append(row["requirement"])
                      d["Reason"].append(row["Reason"])
                      break
              #print(row)
              if index==0:
                  d={}
                  d["question"]=row["Conversation"]
                  d["Answer"]=row["Answer"]
                  d["requirement"]=[]
                  d["requirement"].append(row["requirement"])
                  d["Reason"]=[]
                  if "status" in row:
                    d["status"]=row["status"]
                  d["Reason"].append(row["Reason"])
                  dd.append(d)

   

      flat_data = json_normalize(dd,sep=".",max_level=1)

      flat_data.to_csv(filename, index=False)

      


def main():
    parser = argparse.ArgumentParser(description="Run VulnScanner with specified requirements and output files.")
    parser.add_argument("--requirements_file", type=str, help="Path to the requirements file", required=True)
    parser.add_argument("--output_file", type=str, help="Path to the output file where the report is generated", required=True)
    parser.add_argument("--input_file", type=str, help="Path to the input file where queries are present", required=True)
    parser.add_argument("--model", type=str, help="LLM client model", required=False)
    parser.add_argument("--llm_client", type=str, help="LLM client name", required=False)
    parser.add_argument("--module_name", type=str, help="Name of the module containing the function", required=False)
    parser.add_argument("--function_name", type=str, help="Name of the function to use", required=False)

    args = parser.parse_args()


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
  
    os.environ["REQUIREMENTS_FILE"] = args.requirements_file

    
    # Dynamically import the module and function
    module = importlib.import_module(args.module_name)
    model_function = getattr(module, args.function_name)


    # Set up the LLM client
    g = GroqClient(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))
    set_default_client(g)

    # Create and run the scanner
    v = VulnScanner(input_file=args.input_file,output_file=args.output_file,model_function=model_function)
    v.run_scanner()


if __name__ == "__main__":
    main()


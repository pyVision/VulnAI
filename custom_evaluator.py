
import giskard
import os
import requests
import json
from giskard import Dataset, Model, scan, GiskardClient
from typing import Dict, Optional, Sequence, Tuple
import re
from giskard.llm.client.base import ChatMessage
from giskard.llm.evaluators import RequirementEvaluator
from giskard.llm.evaluators.base import EvaluationResult
import re
import json
from giskard.llm.errors import LLMGenerationError


class CustomEvaluator(RequirementEvaluator):


    def extract_json(self,input_text):
        # Define a regular expression pattern to match a JSON object
        json_pattern = r'\{.*?\}'

        # Search for the first occurrence of the JSON pattern in the input text
        match = re.search(json_pattern, input_text)

        if match:
            # Extract the JSON string from the match
            json_str = match.group(0)

            try:
                # Parse the JSON string to ensure it's valid
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError:
                print("Invalid JSON found.")
                return None
        else:
            print("No JSON object found in the input text.")
            return None

    def _parse_evaluation_output(self, raw_eval: ChatMessage) -> Tuple[bool, Optional[str]]:
        try:
            eval_result = self.extract_json(raw_eval.content)
            return eval_result["eval_passed"], eval_result.get("reason")
        except (AttributeError, KeyError, json.JSONDecodeError) as err:
            raise LLMGenerationError("Could not parse evaluator output") from err
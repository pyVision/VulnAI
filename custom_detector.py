import giskard
import os
import requests
import json
from giskard import Dataset, Model, scan, GiskardClient
from typing import Dict, Optional, Sequence, Tuple
import re
from giskard.llm.client.base import ChatMessage
# Import necessary modules from giskard
from giskard.scanner.llm.llm_harmful_content_detector import LLMHarmfulContentDetector
from giskard.scanner.decorators import detector
from giskard.scanner.scanner import Scanner
from giskard.scanner.issues import Harmfulness, IssueLevel
from typing import Optional, Sequence
from giskard.models.base.model import BaseModel
from giskard.scanner.issues import Issue
from giskard.scanner import logger
from giskard.llm.generators import AdversarialDataGenerator
from giskard.llm.testcase import TestcaseRequirementsGenerator
import pandas as pd
from giskard.datasets.base import Dataset
from giskard.llm.evaluators import RequirementEvaluator
from giskard.llm.evaluators.base import EvaluationResult
from giskard.llm.generators import AdversarialDataGenerator
from giskard.llm.testcase import TestcaseRequirementsGenerator
from giskard.models.base.model import BaseModel
from giskard.testing.tests.llm import test_llm_output_against_requirement
from giskard.scanner.issues import Issue
from giskard.scanner.registry import Detector
from giskard.scanner import logger
from giskard.llm.utils import format_chat_messages


HARMFULNESS_ISSUE_DESCRIPTION = """"""

from giskard.llm.errors import LLMGenerationError

from custom_evaluator import CustomEvaluator


# Define a custom detector using a decorator
@detector("llm_custom_detector", tags=["llm_custom_detector", "harmfulness", "text_generation"])
class CustomDetector(LLMHarmfulContentDetector):
    """
    A custom detector that evaluates the outputs of LLM-based models for harmful content.
    
    This detector uses requirements specified for assessing whether generated content
    is inappropriate, unsafe, or unsuitable for young audiences. It utilizes OpenAI's 
    GPT-4 model for evaluation.

    Attributes:
        num_requirements (int): Number of requirements to evaluate.
        num_samples (int): Number of samples to evaluate per requirement.
        llm_seed (int): Random seed for reproducibility.
    """

    def __init__(self, num_requirements=1, num_samples=1, llm_seed: int = 1729):
        """
        Initialize the custom detector with the number of requirements and samples.

        Args:
            num_requirements (int): Number of requirements to evaluate.
            num_samples (int): Number of samples to evaluate per requirement.
            llm_seed (int): Random seed for reproducibility.
        """
        self.num_requirements = num_requirements
        self.num_samples = num_samples
        self.llm_seed = llm_seed
        self.init_requirements()



    def init_requirements(self):
        """
        Initialize the requirements by reading from a file specified by the environment variable REQUIREMENTS_FILE.

        The file should contain each requirement on a new line.

        Returns:
            list: A list of requirements read from the file.
        """
        # Retrieve the file path from the environment variable
        requirements_file = os.getenv("REQUIREMENTS_FILE")
        
        # Create a list to store each requirement from the file
        requirements_list = []

        # Check if the file path is provided
        if requirements_file is None:
            print("Environment variable REQUIREMENTS_FILE is not set.")
            return requirements_list

        # Try to open the file and read its contents
        try:
            with open(requirements_file, 'r') as file:
                # Read each line from the file and add it to the list
                for line in file:
                    # Strip any whitespace and add to the list
                    requirements_list.append(line.strip())
        except FileNotFoundError:
            print(f"Failed to find the file: {requirements_file}")
        except IOError as e:
            print(f"Error reading the file: {requirements_file}. Error: {e}")

        self.requirement=[]
        self.requirement.extend(requirements_list)
        self.num_requirements = len(self.requirement)
        

    def get_issue_description(self):
        """
        Return the description of the harmfulness issue detected.

        Returns:
            str: Description of the harmfulness issue.
        """
        return HARMFULNESS_ISSUE_DESCRIPTION

    def make_issue(self, model: BaseModel, dataset: Dataset, requirement: str, eval_result: EvaluationResult) -> Issue:
        """
        Create an issue report for a model evaluation based on the provided evaluation results.

        This method generates a detailed issue report containing examples of both failures and successes
        in meeting the specified requirement.

        Parameters:
            model (BaseModel): The machine learning model being evaluated.
            dataset (Dataset): The dataset used for evaluation.
            requirement (str): The requirement that the model is expected to fulfill.
            eval_result (EvaluationResult): The result of the evaluation, containing examples of failures and successes.

        Returns:
            Issue: An Issue object representing the report of the evaluation, including detailed examples and metadata.
        """
        
        # Create a DataFrame from failure examples, including formatted conversation, reason for failure, and status.
        examples = pd.DataFrame(
            [
                {
                    "Conversation": format_chat_messages(ex["sample"].get("conversation", [])),
                    "Reason": ex.get("reason", "No reason provided."),
                    "status": "1",  # Status '1' indicates a failure
                }
                for ex in eval_result.failure_examples
            ]
        )

        # Create a DataFrame from success examples, including formatted conversation, reason for success, and status.
        examples1 = pd.DataFrame(
            [
                {
                    "Conversation": format_chat_messages(ex["sample"].get("conversation", [])),
                    "Reason": ex.get("reason", "No reason provided."),
                    "status": "0",  # Status '0' indicates a success
                }
                for ex in eval_result.success_examples
            ]
        )

        # Concatenate failure and success examples into a single DataFrame if there are any success examples
        if not examples1.empty:
            result = pd.concat([examples, examples1], ignore_index=True)
        else:
            result = examples

        # Create an Issue object with metadata and examples of failures and successes
        ii = Issue(
            model,
            dataset,
            group=self._issue_group,  # Issue group classification
            level=self._issue_level,  # Issue severity level
            description="The model does not satisfy the following requirement: " + requirement,
            examples=result.copy(deep=True),  # Deep copy of the DataFrame to avoid unintended modifications
            meta={
                "metric": "Failing samples",  # Metric name for tracking
                "metric_value": len(examples),  # Number of failing examples
                "domain": requirement,  # The domain or area of the requirement
                "requirement": requirement,  # Requirement description
                "deviation": f"Found {len(examples)} model output{'s' if len(examples) > 1 else ''} out of {len(result)} not meeting the requirement",  # Summary of deviations
                "hide_index": True,  # Option to hide index in the output
            },
            # tests=_generate_output_requirement_tests,  # Optional: Placeholder for generating specific output tests
            taxonomy=self._taxonomy,  # Taxonomy for categorizing the issue
        )

        return ii


    def run(self, model: BaseModel, dataset: Dataset, features=None,llm_client=None) -> Sequence[Issue]:
        """
        Run the detector on a model and dataset to identify harmful content.

        Args:
            model (BaseModel): The model to evaluate.
            dataset (Dataset): The dataset to evaluate against.
            features (optional): Specific features to consider.

        Returns:
            Sequence[Issue]: List of detected issues.
        """
        requirements = self.requirement

        logger.info(f"{self.__class__.__name__}: Evaluating test cases")
        issues = []
        eval_dataset = dataset

        # Loop over each requirement
        for requirement in requirements:
            logger.info(f"{self.__class__.__name__}: Evaluating requirement: {requirement}")

            eval_dataset = dataset

            # Initialize the requirement evaluator
            evaluator = CustomEvaluator([requirement],llm_client=llm_client)

            # Run the model evaluation against each requirement
            eval_result = evaluator.evaluate(model, eval_dataset)

            
            # Append the issues
            issues.append(self.make_issue(model, eval_dataset, requirement, eval_result))

            if eval_result.failed:
                logger.info(
                    f"{self.__class__.__name__}: Test case failed ({len(eval_result.failure_examples)} failed examples)"
                )
            else:
                logger.info(f"{self.__class__.__name__}: Test case passed")

        return issues
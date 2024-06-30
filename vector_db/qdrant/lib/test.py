import dotenv

from llm import LLM
import json
from pprint import pprint

from tqdm import tqdm
from typing import Literal
from fact_check import FACT_CHECK
import os

dotenv.load_dotenv()


class GENERATE_ENHANCE_QUERY(LLM):
    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "/generate_enhance_query.txt"

    N_QUERY_SPLIT = 20

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
        ] = "gpt3",
        temperature=0,
        max_tokens=3000,
        top_p=0,
    ) -> None:
        self.data = ALL_ENHANCE_QUERY([])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
        )

    def split_number_with_interval(self, total: int):
        return [self.N_QUERY_SPLIT] * (total // self.N_QUERY_SPLIT) + [
            total % self.N_QUERY_SPLIT
        ]

    def generate_enhance_query_once(self, number_of_query: int):
        keywords = "\n".join(self.data.get_keywords())

        self.prompt = self.prompt.replace("<n_query>", str(number_of_query))
        self.prompt = self.prompt.replace("<keywords>", keywords)

        result = ALL_ENHANCE_QUERY(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )["result"]
        )
        self.data.data.extend(result.data)

    def generate_enhance_query(self):
        self.prompt = self.prompt.replace("<technology>", self.technology)
        self.prompt = self.prompt.replace("<query>", self.query)
        for number_of_query in tqdm(
            self.split_number_with_interval(self.number_of_query), desc="GENERATE QUERY"
        ):
            prompt = self.prompt
            self.generate_enhance_query_once(number_of_query=number_of_query)
            self.save_data()
            self.prompt = prompt

    def generate(self, technology: str, query: str, number_of_query: int):
        self.technology = technology
        self.query = query
        self.number_of_query = number_of_query
        self.generate_enhance_query()

import dotenv
import json
from pprint import pprint
from llm import LLM
import pandas as pd
from generate_description import GENERATE_TECHNOLOGY_DESCRIPTION, ALL_MARKET_DESCRIPTION
import os
from typing import Literal

dotenv.load_dotenv()


class GENERATE_RECOMMEND_INDUSTRY(ALL_MARKET_DESCRIPTION, LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "recommend_market.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_recommend_market"

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
        max_tokens=2000,
        top_p=0,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
        )

    def show_data(self):
        return {
            f"{item.japanese_name}({item.name}) : {item.japanese_description}"
            for item in self.data
        }

    def save_data(self, technology: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{technology}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_recommend(self):
        generated_recommend = self.get_response(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        return json.loads(generated_recommend)["result"]

    def get_recommend(
        self,
        technology: str,
        description: str,
        number_of_markets: int,
    ) -> list[dict]:
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)
        self.prompt = self.prompt.replace("<number_of_markets>", str(number_of_markets))

        result = self.generate_recommend()
        ALL_MARKET_DESCRIPTION.__init__(self, result)

    def recommend(
        self,
        technology: str,
        description: str,
        number_of_markets: int,
    ) -> list[dict]:
        """

        Args:
            technology (str): Technology Keyword
            description (str): Technology Description

        Returns:
            list[dict]:
            [
                {
                    "name": "...",
                    "english_name": "...",
                    "description": "..."
                    "japanese_description": "..."
                }
            ]
        """
        self.get_recommend(
            technology=technology,
            description=description,
            number_of_markets=number_of_markets,
        )
        self.save_data(technology=technology)

        return self.get_data()

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = json.load(f)
            ALL_MARKET_DESCRIPTION.__init__(self, data)
            f.close()


def test_gen():
    query = "LED"
    print("QUERY:", query)
    description_generator = GENERATE_TECHNOLOGY_DESCRIPTION()

    file_path = "./data/generated_technology_description/LED_claude1.json"
    description_generator.read_data(file_path)

    description = description_generator.get_dict()["description"]

    print("GENERATED DESCRIPTION:", description)

    recommend_generator = GENERATE_RECOMMEND_INDUSTRY(use_model="claude2")
    recommend = recommend_generator.recommend(
        technology=query,
        description=description,
        number_of_markets=6,
    )
    print("RECOMMEND:")
    pprint(recommend)


def test_read():
    file_path = "./data/generated_recommend_market/LED_claude2.json"
    recommend_generator = GENERATE_RECOMMEND_INDUSTRY(use_model="claude2")
    recommend_generator.read_data(file_path)
    print("RECOMMEND:")
    pprint(recommend_generator.get_data())


def main():
    test_gen()
    test_read()


if __name__ == "__main__":
    main()

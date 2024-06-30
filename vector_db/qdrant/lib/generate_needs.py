import dotenv

from llm import LLM
import json
from pprint import pprint

from generate_description import (
    GENERATE_TECHNOLOGY_DESCRIPTION,
    GENERATE_MARKET_DESCRIPTION,
    MARKET_DESCRIPTION,
    ALL_MARKET_DESCRIPTION,
)
from recommend_market import GENERATE_RECOMMEND_INDUSTRY
from typing import Literal
import os

dotenv.load_dotenv()


class NEEDS:

    def __init__(
        self,
        title: str,
        japanese_title: str,
        description: str = "",
        japanese_description: str = "",
        objective: Literal["basic", "escape", "enhance"] = "",
    ):
        self.title = title
        self.japanese_title = japanese_title
        self.description = description
        self.japanese_description = japanese_description
        self.objective = str(objective).lower()

    def get_dict(self):
        """
        Returns:
            {
                title: Needs Title
                japanese_title: 日本語Needs名
                description: Needs Description
                japanese_description: 日本語Needs概要
                objective: basic | escape | enhance
            }
        """
        return {
            "title": self.title,
            "japanese_title": self.japanese_title,
            "description": self.description,
            "japanese_description": self.japanese_description,
            "objective": self.objective,
        }


class MARKET_NEEDS:

    def __init__(self, market_needs: list):
        self.data = list[NEEDS](self.read_market_needs_list(market_needs))

    def read_market_needs_list(self, market_needs: list):
        return [
            NEEDS(
                title=needs["title"],
                japanese_title=needs["japanese_title"],
                description=needs["description"],
                japanese_description=needs["japanese_description"],
                objective=needs["objective"],
            )
            for needs in market_needs
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]


class GENERATE_NEEDS(LLM):
    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generate_needs.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_needs"

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
        self.data: dict[str, MARKET_NEEDS] = {}
        self.prompt = ""

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
        )

    def get_data(self):
        return {key: value.get_data() for key, value in self.data.items()}

    def save_data(self, technology: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{technology}_{self.use_model}.json", "w+"
        ) as f:
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_one_needs(self, market: MARKET_DESCRIPTION) -> list[dict]:
        market_string = f"{market.name}({market.japanese_description}): {market.description}({market.description})"
        prompt, self.prompt = (
            self.prompt,
            self.prompt.replace("<market>", market_string),
        )

        generated_needs = self.get_response(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        self.prompt = prompt

        return json.loads(generated_needs)["result"]

    def generate_market_needs(self, technology: str, markets: ALL_MARKET_DESCRIPTION):
        for market in markets.data:
            generated_needs = self.generate_one_needs(market)
            self.data[market.name] = MARKET_NEEDS(generated_needs)
            self.save_data(technology=technology)

    def generate_needs(
        self,
        technology: str,
        description: str,
        markets: list[MARKET_DESCRIPTION],
        number_of_needs: int,
    ):
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)
        self.prompt = self.prompt.replace("<number_of_needs>", str(number_of_needs))

        self.generate_market_needs(
            technology=technology,
            markets=markets,
        )

        return self.get_data()

    def generate(
        self,
        technology: str,
        description: str,
        markets: list,
        number_of_needs=6,
    ):
        return self.generate_needs(
            technology=technology,
            description=description,
            markets=ALL_MARKET_DESCRIPTION(markets),
            number_of_needs=number_of_needs,
        )

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = dict(json.load(f))
            self.data = {
                market: MARKET_NEEDS(needs_list) for market, needs_list in data.items()
            }
            f.close()


def test_gen_market_needs():
    query = "LED"
    # print("QUERY:", query)

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    file_path = "./data/generated_technology_description/LED_claude1.json"
    description_gen.read_data(file_path)
    description = description_gen.get_dict()["description"]

    industry_recommend = GENERATE_RECOMMEND_INDUSTRY(use_model="claude1")
    file_path = "./data/generated_recommend_market/LED_claude2.json"
    industry_recommend.read_data(file_path)
    markets = industry_recommend.get_data()[3:]

    print("INDUSTRY")
    pprint(markets)

    additional_market = ["ウェルビーイング市場", "医療"]
    market_description_gen = GENERATE_MARKET_DESCRIPTION(use_model="claude1")
    market_description_gen.generate(markets=additional_market)
    additional_market = market_description_gen.get_data()
    markets.extend(additional_market)

    needs_gen = GENERATE_NEEDS(use_model="claude1", max_tokens=4000, top_p=0)
    needs_gen.generate(
        technology=query,
        description=description,
        markets=markets,
        number_of_needs=10,
    )


def test_read_generated_need():
    file_path = "./data/generated_needs/LED_claude1.json"
    needs_gen = GENERATE_NEEDS()
    needs_gen.read_data(file_path=file_path)
    pprint(needs_gen.get_data())


def test():

    query = "SAF"

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    description_gen.generate(query)
    description_result = description_gen.get_dict()
    description = f"{description_result['description']}({description_result['japanese_description']})"
    pprint(description)

    industry_recommend = GENERATE_RECOMMEND_INDUSTRY(use_model="claude1")
    industry_recommend.recommend(
        technology=query,
        description=description,
        number_of_markets=6,
    )
    markets = industry_recommend.get_data()

    print("INDUSTRY")
    pprint(industry_recommend.show_data())

    # additional_market = ["ウェルビーイング市場", "医療"]
    market_description_gen = GENERATE_MARKET_DESCRIPTION(use_model="claude1")
    market_description_gen.generate(markets=additional_market)
    additional_market = market_description_gen.get_data()
    markets.extend(additional_market)

    needs_gen = GENERATE_NEEDS(use_model="gpt3", max_tokens=3000, top_p=0)
    needs_gen.generate(
        technology=query,
        description=description,
        markets=markets,
        number_of_needs=5,
    )


def main():
    # test_gen_market_needs()
    # test_read_generated_need()
    test()


if __name__ == "__main__":
    main()

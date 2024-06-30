import dotenv
import os
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
from generate_needs import GENERATE_NEEDS, NEEDS, MARKET_NEEDS

from typing import Literal


dotenv.load_dotenv()


class FUNCTION:

    def __init__(
        self,
        title: str,
        japanese_title: str,
        description: str,
        japanese_description: str,
    ):
        self.title = title
        self.japanese_title = japanese_title
        self.description = description
        self.japanese_description = japanese_description

    def get_dict(self):
        """
        Returns:
            {
                title: Function Title
                japanese_title: 日本語Function名
                description: Function Description
                japanese_description: 日本語Function概要
            }
        """
        return {
            "title": self.title,
            "japanese_title": self.japanese_title,
            "description": self.description,
            "japanese_description": self.japanese_description,
        }


class TECHNOLOGY_FUNCTION:

    def __init__(self, technology_functions: list):
        self.data = list[FUNCTION](
            self.read_technology_functions_list(technology_functions)
        )

    def read_technology_functions_list(self, technology_functions: list):
        return [
            FUNCTION(
                title=functions["title"],
                japanese_title=functions["japanese_title"],
                description=functions["description"],
                japanese_description=functions["japanese_description"],
            )
            for functions in technology_functions
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]


class GENERATE_FUNCTIONS(TECHNOLOGY_FUNCTION, LLM):
    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generate_functions.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_functions"

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

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
        )

    def save_data(self, technology: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{technology}_{self.use_model}.json", "w+"
        ) as f:
            pprint(self.get_data())
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_functions(
        self,
        technology: str,
        description: str,
        number_of_functions: int,
    ):
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)
        self.prompt = self.prompt.replace(
            "<number_of_functions>", str(number_of_functions)
        )

        generated_function = json.loads(
            self.get_response(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        )["result"]

        TECHNOLOGY_FUNCTION.__init__(self, technology_functions=generated_function)
        self.save_data(technology=technology)

    def generate(
        self,
        technology: str,
        description: str,
        number_of_functions=20,
    ):
        self.generate_functions(
            technology=technology,
            description=description,
            number_of_functions=number_of_functions,
        )

        return self.get_data()

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = json.load(f)
            TECHNOLOGY_FUNCTION.__init__(self, technology_functions=data)
            f.close()


class GENERATE_INDUSTRY_FUNCTIONS(LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generate_market_functions.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generate_market_functions"

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

        self.data: dict[str, TECHNOLOGY_FUNCTION] = {}
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
            pprint(self.get_data())
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_market_function(self, market: MARKET_DESCRIPTION) -> list[dict]:
        market_string = f"{market.name}({market.japanese_name}): {market.japanese_description}({market.description})"
        prompt, self.prompt = (
            self.prompt,
            self.prompt.replace("<market>", market_string),
        )

        generated_functions = json.loads(
            self.get_response(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        )["result"]

        self.prompt = prompt

        return generated_functions

    def generate_market_functions(
        self,
        technology: str,
        description: str,
        number_of_functions: int,
        markets: ALL_MARKET_DESCRIPTION,
    ):
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)
        self.prompt = self.prompt.replace(
            "<number_of_functions>", str(number_of_functions)
        )

        for market in markets.data:
            self.data[market.name] = TECHNOLOGY_FUNCTION(
                self.generate_market_function(market)
            )
            self.save_data(technology=technology)

    def generate(
        self,
        technology: str,
        description: str,
        markets: list,
        number_of_functions=10,
    ):
        self.generate_market_functions(
            technology=technology,
            description=description,
            markets=ALL_MARKET_DESCRIPTION(markets),
            number_of_functions=number_of_functions,
        )

        return self.get_data()

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = dict(json.load(f))
            self.data = {
                market: TECHNOLOGY_FUNCTION(function_list)
                for market, function_list in data.items()
            }
            f.close()


class GENERATE_INDUSTRY_NEEDS_FUNCTIONS(LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generate_market_needs_functions.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generate_market_needs_functions"

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

        self.data: dict[str, TECHNOLOGY_FUNCTION] = {}

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
            pprint(self.get_data())
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_market_needs_function(
        self,
        market: MARKET_DESCRIPTION,
        market_needs: list[NEEDS],
    ) -> list[dict]:
        market_string = f"{market.name}({market.japanese_name}): {market.japanese_description}({market.description})"

        market_needs_string = "\n".join(
            [
                f"{index+1}. {item.title}({item.description}): {item.description}({item.japanese_description})"
                for index, item in enumerate(market_needs)
            ]
        )

        prompt = self.prompt
        self.prompt = self.prompt.replace("<market>", market_string)
        self.prompt = self.prompt.replace("<market_needs>", market_needs_string)

        print(self.prompt)

        generated_functions = self.get_response(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        self.prompt = prompt

        return json.loads(generated_functions)["result"]

    def generate_market_needs_functions(
        self,
        technology: str,
        description: str,
        number_of_functions: int,
        markets: ALL_MARKET_DESCRIPTION,
        market_needs: dict[str, MARKET_NEEDS],
    ):
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)
        self.prompt = self.prompt.replace(
            "<number_of_functions>", str(number_of_functions)
        )
        for market in markets.data:
            self.data[market.name] = TECHNOLOGY_FUNCTION(
                self.generate_market_needs_function(
                    market=market,
                    market_needs=market_needs[market.name].data,
                )
            )
            self.save_data(technology=technology)

    def generate(
        self,
        technology: str,
        description: str,
        markets: list[str],
        market_needs: dict[str, list],
        number_of_functions=20,
    ):
        self.generate_market_needs_functions(
            technology=technology,
            description=description,
            markets=ALL_MARKET_DESCRIPTION(markets),
            market_needs={
                market: MARKET_NEEDS(needs) for market, needs in market_needs.items()
            },
            number_of_functions=number_of_functions,
        )

        return self.get_data()

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = dict(json.load(f))
            self.data = {
                market: TECHNOLOGY_FUNCTION(function_list)
                for market, function_list in data.items()
            }
            f.close()


def test_gen_function():
    query = "LED"
    query = "Sustainable Aviation Fuel"

    # print("QUERY:", query)

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    # file_path = "./data/generated_technology_description/LED_claude1.json"
    # description_gen.read_data(file_path)
    description_gen.generate(query)
    description = description_gen.get_dict()["description"]

    market_recommend = GENERATE_RECOMMEND_INDUSTRY(use_model="claude1")
    file_path = "./data/generated_recommend_market/LED_claude2.json"
    market_recommend.read_data(file_path)
    markets = market_recommend.get_data()[3:]

    print("INDUSTRY")
    pprint(markets)

    additional_market = ["ウェルビーイング市場", "医療"]
    market_description_gen = GENERATE_MARKET_DESCRIPTION(use_model="claude1")
    file_path = (
        "./data/generated_market_description/ウェルビーイング市場,医療_claude1.json"
    )
    market_description_gen.read_data(file_path)
    additional_market = market_description_gen.get_data()

    markets.extend(additional_market)
    file_path = "./data/generated_needs/LED_claude1.json"
    needs_gen = GENERATE_NEEDS()
    needs_gen.read_data(file_path)
    needs = needs_gen.get_data()
    print("INDUSTRY")
    pprint(needs)

    function_gen = GENERATE_FUNCTIONS(
        use_model="claude2",
        max_tokens=4000,
    )
    function_gen.generate(
        technology=query,
        description=description,
        number_of_functions=10,
    )
    pprint(function_gen.get_data())


def test_read_gen_function():

    file_path = "./data/generated_functions/LED_claude1.json"
    function_gen = GENERATE_FUNCTIONS()
    function_gen.read_data(file_path)
    pprint(function_gen.get_data())


def test_generate_market_function():
    query = "LED"
    # print("QUERY:", query)

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    file_path = "./data/generated_technology_description/LED_claude1.json"
    description_gen.read_data(file_path)
    description = description_gen.get_dict()["description"]

    market_recommend = GENERATE_RECOMMEND_INDUSTRY(use_model="claude1")
    file_path = "./data/generated_recommend_market/LED_claude2.json"
    market_recommend.read_data(file_path)
    markets = market_recommend.get_data()[3:]

    additional_market = ["ウェルビーイング市場", "医療"]
    market_description_gen = GENERATE_MARKET_DESCRIPTION(use_model="claude1")
    file_path = (
        "./data/generated_market_description/ウェルビーイング市場,医療_claude1.json"
    )
    market_description_gen.read_data(file_path)
    additional_market = market_description_gen.get_data()

    markets.extend(additional_market)
    file_path = "./data/generated_needs/LED_claude1.json"
    needs_gen = GENERATE_NEEDS()
    needs_gen.read_data(file_path)
    needs = needs_gen.get_data()

    function_gen = GENERATE_INDUSTRY_FUNCTIONS(
        use_model="claude1",
        max_tokens=3000,
    )
    function_gen.generate(
        technology=query,
        description=description,
        markets=markets,
        number_of_functions=10,
    )


def test_read_gen_market_function():

    file_path = "./data/generate_market_functions/LED_claude1.json"
    function_gen = GENERATE_INDUSTRY_FUNCTIONS()
    function_gen.read_data(file_path)
    pprint(function_gen.get_data())


def test_gen_market_needs_function():
    query = "LED"
    # print("QUERY:", query)

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    file_path = "./data/generated_technology_description/LED_claude1.json"
    description_gen.read_data(file_path)
    description = description_gen.get_dict()["description"]

    market_recommend = GENERATE_RECOMMEND_INDUSTRY(use_model="claude1")
    file_path = "./data/generated_recommend_market/LED_claude1.json"
    market_recommend.read_data(file_path)
    markets = market_recommend.get_data()

    additional_market = ["ウェルビーイング市場", "医療"]
    market_description_gen = GENERATE_MARKET_DESCRIPTION(use_model="claude1")
    file_path = (
        "./data/generated_market_description/ウェルビーイング市場,医療_claude1.json"
    )
    market_description_gen.read_data(file_path)
    additional_market = market_description_gen.get_data()

    markets.extend(additional_market)
    file_path = "./data/generated_needs/LED_claude1.json"
    needs_gen = GENERATE_NEEDS()
    needs_gen.read_data(file_path)
    needs = needs_gen.get_data()

    function_gen = GENERATE_INDUSTRY_NEEDS_FUNCTIONS(
        use_model="claude2",
    )
    function_gen.generate(
        technology=query,
        description=description,
        markets=markets,
        market_needs=needs,
        number_of_functions=20,
    )


def test_read_gen_market_need_function():

    file_path = "./data/generate_market_needs_functions/LED_claude1.json"
    function_gen = GENERATE_INDUSTRY_NEEDS_FUNCTIONS()
    function_gen.read_data(file_path)
    pprint(function_gen.get_data())


def main():
    test_gen_function()
    # test_read_gen_function()
    # test_generate_market_function()
    # test_read_gen_market_function()
    test_gen_market_needs_function()
    # test_read_gen_market_need_function()
    pass


if __name__ == "__main__":
    main()

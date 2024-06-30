import dotenv

from llm import LLM
import json
from pprint import pprint
from typing import Literal
import os

dotenv.load_dotenv()


class TECHNOLOGY_DESCRIPTION:
    def __init__(
        self,
        name: str,
        japanese_name: str,
        description: str,
        japanese_description: str,
    ) -> None:
        self.name = name
        self.japanese_name = japanese_name
        self.description = description
        self.japanese_description = japanese_description

    def get_dict(self):
        """

        Returns:
            {
                name: Technology Name,
                japanese_name: 技術名,
                japanese_description: 日本語の概要,
                "description": Description,
            }
        """
        return {
            "name": self.name,
            "japanese_name": self.japanese_name,
            "japanese_description": self.japanese_description,
            "description": self.description,
        }


class ALL_TECHNOLOGY_DESCRIPTION:

    def __init__(self, technology_descriptions: list):
        self.data = list[TECHNOLOGY_DESCRIPTION](
            self.read_technology_description_list(
                technology_descriptions=technology_descriptions
            )
        )

    def read_technology_description_list(self, technology_descriptions: list):
        return [
            TECHNOLOGY_DESCRIPTION(
                name=item["name"],
                japanese_name=item["japanese_name"],
                description=item["description"],
                japanese_description=item["japanese_description"],
            )
            for item in technology_descriptions
        ]

    def get_data(self) -> list[TECHNOLOGY_DESCRIPTION]:
        """

        Returns: list
            [{
                name: Technology Name,
                japanese_name: 技術名,
                japanese_description: 日本語の概要,
                "description": Description,
            }]
        """

        return [item.get_dict() for item in self.data]


class MARKET_DESCRIPTION:
    def __init__(
        self,
        name: str,
        japanese_name: str,
        description: str,
        japanese_description: str,
    ) -> None:
        self.name = name
        self.japanese_name = japanese_name
        self.description = description
        self.japanese_description = japanese_description

    def get_dict(self):
        """

        Returns:
            {
                name: Market Name,
                japanese_name: 市場名,
                japanese_description: 日本語の概要,
                "description": Description,
            }
        """
        return {
            "name": self.name,
            "japanese_name": self.japanese_name,
            "japanese_description": self.japanese_description,
            "description": self.description,
        }


class ALL_MARKET_DESCRIPTION:

    def __init__(self, market_descriptions: list):
        self.data = list[MARKET_DESCRIPTION](
            self.read_market_description_list(market_descriptions=market_descriptions)
        )

    def read_market_description_list(self, market_descriptions: list):
        return [
            MARKET_DESCRIPTION(
                name=item["name"],
                japanese_name=item["japanese_name"],
                description=item["description"],
                japanese_description=item["japanese_description"],
            )
            for item in market_descriptions
        ]

    def get_data(self) -> list[MARKET_DESCRIPTION]:
        """

        Returns:
            [{
                name: Market Name,
                japanese_name: 市場名,
                japanese_description: 日本語の概要,
                "description": Description,
            },...]
        """
        return [item.get_dict() for item in self.data]


class GENERATE_TECHNOLOGY_DESCRIPTION(TECHNOLOGY_DESCRIPTION, LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generate_technology_description.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_technology_description"

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
        ] = "claude1",
        temperature=0,
        max_tokens=500,
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

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = json.load(f)
            TECHNOLOGY_DESCRIPTION.__init__(
                self,
                name=data["name"],
                japanese_name=data["japanese_name"],
                japanese_description=data["japanese_description"],
                description=data["description"],
            )
            f.close()

    def save_data(self, technology: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{technology}_{self.use_model}.json", "w+"
        ) as f:
            json.dump(self.get_dict(), f, ensure_ascii=False)
            f.close()

    def generate_description(self, technology: str) -> None:
        """
        Returns:
            {
                "japanese_description": "日本語の概要",
                "description": "English Description",
            }
        """
        self.prompt = self.prompt.replace("<technology>", technology)
        generated_description = json.loads(
            self.get_response(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        )

        TECHNOLOGY_DESCRIPTION.__init__(
            self,
            name=generated_description["name"],
            japanese_name=generated_description["japanese_name"],
            japanese_description=generated_description["japanese_description"],
            description=generated_description["description"],
        )

    def generate(self, technology: str) -> json:
        """

        Returns:
            {
                name: Technology Name,
                japanese_name: 技術名,
                japanese_description: 日本語の概要,
                "description": Description,
            }
        """
        self.generate_description(technology=technology)
        self.save_data(technology=technology)

        return self.get_dict()


class GENERATE_MARKET_DESCRIPTION(ALL_MARKET_DESCRIPTION, LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "generated_market_description.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_market_description"

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

    def save_data(self, market_string: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{market_string}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def generate_description(self, markets: list[str]):
        market_string = "\n".join(markets)
        self.prompt = self.prompt.replace("<market>", market_string)
        generated_market_description = dict(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )
        )["result"]

        ALL_MARKET_DESCRIPTION.__init__(
            self,
            market_descriptions=generated_market_description,
        )

    def generate(self, markets: list[str]):
        self.generate_description(markets=markets)
        self.save_data(market_string=",".join(markets))

        return self.get_data()

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = json.load(f)
            ALL_MARKET_DESCRIPTION.__init__(self, data)
            f.close()


def test_gen_technology_description():
    client = GENERATE_TECHNOLOGY_DESCRIPTION(
        use_model="claude1",
    )

    query = "GNN"

    client.generate(query)
    pprint(client.get_dict())


def test_read_technology_description():
    client = GENERATE_TECHNOLOGY_DESCRIPTION(
        use_model="claude1",
    )

    file_path = "./data/generated_technology_description/GNN_claude1.json"
    client.read_data(file_path)
    pprint(client.get_dict())


def test_gen_market_description():
    client = GENERATE_MARKET_DESCRIPTION(
        use_model="claude1",
        temperature=0,
    )

    query = ["ウェルビーイング", "医療"]

    client.generate(query)
    pprint(client.get_data())


def test_read_market_description():
    client = GENERATE_MARKET_DESCRIPTION(
        use_model="claude1",
    )

    file_path = "./data/generated_market_description/ウェルビーイング,医療_claude1.json"
    client.read_data(file_path)
    pprint(client.get_data())


def test_tech_multiple_models():
    query = "Causal Inference"
    models = [
        "gpt3",
        "gpt4",
        "claude1",
        "claude2",
        "claude2-1",
    ][4:]
    for model in models:
        client = GENERATE_TECHNOLOGY_DESCRIPTION(use_model=model)
        client.generate(technology=query)
        print(model)
        pprint(client.get_dict())


def main():
    # test_gen_technology_description()
    # test_read_technology_description()
    # test_gen_market_description()
    # test_read_market_description()

    # test_tech_multiple_models()
    pass


if __name__ == "__main__":
    main()

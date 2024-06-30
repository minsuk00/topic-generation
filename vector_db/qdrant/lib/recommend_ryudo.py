import dotenv

from llm import LLM
import json
from pprint import pprint
import os
from generate_description import (
    GENERATE_TECHNOLOGY_DESCRIPTION,
    ALL_TECHNOLOGY_DESCRIPTION,
)
from typing import Literal


dotenv.load_dotenv()


class GENERATE_RECOMMEND_RYUDO(LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "recommend_ryudo.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_technology_ryudo"

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
        max_tokens=1000,
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

    def get_data(self):
        return {key: value.get_data() for key, value in self.data.items()}

    def read_data(self, file_path: str):
        with open(file_path, "r+") as f:
            data = dict(json.load(f))
            self.data = {
                key: ALL_TECHNOLOGY_DESCRIPTION(technology_descriptions)
                for key, technology_descriptions in data.items()
            }
            f.close()

    def save_data(self, technology: str):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{technology}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.get_data(), f, ensure_ascii=False)
            f.close()

    def recommend_ryudo(self, technology: str, description="None"):
        self.prompt = self.prompt.replace("<technology>", technology)
        self.prompt = self.prompt.replace("<description>", description)

        generated_recommend = dict(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )
        )
        self.data = {
            key: ALL_TECHNOLOGY_DESCRIPTION(technology_descriptions)
            for key, technology_descriptions in generated_recommend.items()
        }

    def recommend(self, technology: str, description: str):
        self.result = self.recommend_ryudo(
            technology=technology,
            description=description,
        )
        self.save_data(technology=technology)
        return self.get_data()


def test_gen():
    query = "GNN"
    print("QUERY:", query)

    file_path = "./data/generated_technology_description/GNN_claude1.json"

    description_gen = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude1")
    description_gen.read_data(file_path)
    description = description_gen.get_dict()["description"]

    recommend_ryudo = GENERATE_RECOMMEND_RYUDO(
        use_model="claude1",
    )

    recommend_ryudo.recommend(technology=query, description=description)
    pprint(recommend_ryudo.get_data())


def test_read():
    query = "GNN"
    print("QUERY:", query)

    file_path = "./data/generated_technology_ryudo/GNN_claude1.json"

    recommend_ryudo = GENERATE_RECOMMEND_RYUDO()

    recommend_ryudo.read_data(file_path)
    pprint(recommend_ryudo.get_data())


def main():
    # test_gen()
    # test_read()
    pass


if __name__ == "__main__":
    main()

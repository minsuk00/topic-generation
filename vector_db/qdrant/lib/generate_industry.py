import dotenv
import json
from llm import LLM
from tqdm import tqdm
import pandas as pd
import os
from typing import Literal

dotenv.load_dotenv()


class INDUSTRY_DATA:
    INDUSTRY_DESCRIPTION_FILE = "./data/industries_description.json"
    INDUSTRY_DATA_FILE = "./data/industries.json"

    def __init__(self) -> None:
        self.read_industries()

    def read_industries(self):
        data = json.load(open(self.INDUSTRY_DATA_FILE))["data"]
        self.industries = {
            major_industry: list(data.keys()) for major_industry, data in data.items()
        }

        self.industry_description = dict(
            json.load(open(self.INDUSTRY_DESCRIPTION_FILE))
        )

        major_industry = list(self.industries.keys())
        minor_industry = [item for items in self.industries.values() for item in items]
        self.prompt = self.prompt.replace(
            "<industry>", f"{','.join(major_industry)},{','.join(minor_industry)}"
        )


class GENERATE_INDUSTRY_DESCRIPTION(INDUSTRY_DATA, LLM):

    PROMPT = "./prompt/generated_industry_description.txt"

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
        top_p=1,
    ) -> None:

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        INDUSTRY_DATA.__init__(self)
        LLM.__init__(
            self,
            prompt_file=self.PROMPT_FILE,
            use_model=use_model,
        )

    def save_industry_description(self):
        df = pd.DataFrame()
        df["Japanes_Industry"] = self.industry_description.keys()
        df["Industry"] = [item["name"] for item in self.industry_description.values()]
        df["Description"] = [
            item["description"] for item in self.industry_description.values()
        ]
        df["Japanese_Description"] = [
            item["japanese_description"] for item in self.industry_description.values()
        ]

        # df.to_csv("./data/industries.csv", index=False)

    def generate_industry_description(self, industry_string: str):
        prompt = self.prompt
        self.prompt = self.prompt.replace("<industry>", industry_string)
        generated_industry_description = self.get_response(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        self.prompt = prompt

        return dict(json.loads(generated_industry_description))

    def generate(self, update=False):
        for major, sub in tqdm(
            self.industries.items(), desc="Generated Industry Description"
        ):
            if not update and major in self.industry_description:
                continue

            query = f"{major}: {','.join(sub)}"
            self.generate_industry_description(industry_string=query)
            generated_industry_description = self.get_response(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            self.industry_description.update(generated_industry_description)
            with open("./data/industries_description.json", "w+") as f:
                pass
                # json.dump(self.industry_description, f, ensure_ascii=False)

            self.save_industry_description()

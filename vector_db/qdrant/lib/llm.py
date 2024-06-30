import os
import openai
import anthropic
from openai import ChatCompletion
import boto3
import json
from typing import Literal
import re
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()


class BEDROCK:
    CLAUDE2 = "anthropic.claude-v2"
    CLAUDE1 = "anthropic.claude-instant-v1"
    CLAUDE21 = "anthropic.claude-v2:1"
    CLAUDE3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_FAMILY = [CLAUDE2, CLAUDE1, CLAUDE21]
    CLAUDE3_FAMILY = [CLAUDE3_HAIKU, CLAUDE3_SONNET]

    TITAN_LITE = "amazon.titan-text-lite-v1"
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_FAMILY = [TITAN_LITE, TITAN_EXPRESS]

    JURASSIC_ULTRA = "ai21.j2-ultra-v1"
    JURASSIC_FAMILY = [JURASSIC_ULTRA]

    def __init__(self, model: str) -> None:
        self.modelId = model
        self.bedrock = boto3.client(service_name="bedrock-runtime")

    def claude_body(self, query: str, max_tokens=256, temperature=0.0, top_p=0.0):
        return json.dumps(
            {
                "prompt": f"Human:{query} Assistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    def titan_body(self, query: str, max_tokens=256, temperature=0, top_p=0):
        return json.dumps(
            {
                "inputText": f"{query}",
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                },
            }
        )

    def jurassic_body(self, query: str, max_tokens=256, temperature=0, top_p=0):
        return json.dumps(
            {
                "prompt": query + "\n---\n{contents}",
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
        )

    def claude3_body(self, query: str, max_tokens=256, temperature=0, top_p=0):
        print(query)
        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            }
        )

    def get_body(self, query: str, max_tokens=256, temperature=0.0, top_p=0.0):
        if self.modelId in self.CLAUDE_FAMILY:
            return self.claude_body(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif self.modelId in self.TITAN_FAMILY:
            return self.titan_body(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif self.modelId in self.JURASSIC_FAMILY:
            return self.jurassic_body(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif self.modelId in self.CLAUDE3_FAMILY:
            return self.claude3_body(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    def get_result(self, response: str):
        response_body = json.loads(response.get("body").read())
        if self.modelId in self.CLAUDE_FAMILY:
            return response_body.get("completion")
        elif self.modelId in self.TITAN_FAMILY:
            return response_body["results"][0].get("outputText").strip()
        elif self.modelId in self.JURASSIC_FAMILY:
            print(response_body["completions"][0])
            return response_body["completions"][0]["data"]["text"]
        elif self.modelId in self.CLAUDE3_FAMILY:
            print(response_body["content"][0])
            return response_body["content"][0]["text"]

    def get_response(self, query: str, max_tokens=256, temperature=0.0, top_p=0.0) -> str:

        body = self.get_body(
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.modelId,
            accept="application/json",
            contentType="application/json",
        )

        return self.get_result(response=response)


class Anthropic:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = anthropic.Client(api_key=os.environ.get(os.environ.get("ANTHROPIC_API_KEY")))
        self.usage_in = 0
        self.usage_out = 0

    def create_query(self, query: str):
        return [
            {"role": "user", "content": query},
        ]

    def get_response(
        self,
        query: str,
        temperature=0,
        max_tokens=256,
        top_p=0,
    ):
        response = self.client.messages.create(
            model=self.model,
            messages=self.create_query(query=query),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        self.usage_in = response.usage.input_tokens
        self.usage_out = response.usage.output_tokens
        return response.content[0].text


class OpenAIBase:
    def __init__(
        self,
        model: str,
        client: openai.OpenAI | openai.AzureOpenAI,
        base: Literal["azure", "openai", "pplx"],
    ):
        self.model = model
        self.client = client
        self.base = base
        self.usage_in = 0
        self.usage_out = 0

    def create_openai_query(self, query: str):
        return [
            {"role": "user", "content": query},
        ]

    def get_response(
        self,
        query: str,
        temperature=0,
        max_tokens=256,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        if self.base == "pplx":
            presence_penalty = 0.1  # must presence_penalty > 0
            frequency_penalty = None

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.create_openai_query(query=query),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self.usage_in = response.usage.prompt_tokens
        self.usage_out = response.usage.completion_tokens
        return response.choices[0].message.content


class AzureOpenAI(OpenAIBase):
    def __init__(self, model: str) -> None:
        client = openai.AzureOpenAI(
            api_key=os.environ.get("AZURE_API_KEY"),
            api_version=os.environ.get("AZURE_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        )
        super().__init__(client=client, model=model, base="azure")


class OpenAI(OpenAIBase):
    def __init__(self, model: str) -> None:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        super().__init__(client=client, model=model, base="opeain")


class PPLX_AI(OpenAIBase):
    def __init__(self, model: str) -> None:
        client = openai.OpenAI(api_key=os.environ.get("PPLX_API_KEY"), base_url="https://api.perplexity.ai")
        super().__init__(client=client, model=model, base="pplx")


class LLM:
    GPT_3_OPENAI = "gpt-3.5-turbo-0125"
    GPT_4_OPENAI = "gpt-4-0125-preview"
    GPT_3_AZURE = "gpt-35-turbo"
    GPT_4_AZURE = "gpt-4"

    CLAUDE2 = "anthropic.claude-v2"
    CLAUDE1 = "anthropic.claude-instant-v1"
    CLAUDE21 = "anthropic.claude-v2:1"
    CLAUDE3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"

    TITAN_LITE = "amazon.titan-text-lite-v1"
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    JURASSIC_ULTRA = "ai21.j2-ultra-v1"

    PPLX_SMALL_CHAT = "sonar-small-chat"
    PPLX_SMALL_ONLINE = "sonar-small-online"
    PPLX_MIDIUM_CHAT = "sonar-medium-chat"
    PPLX_MIDIUM_ONLINE = "sonar-medium-online"
    PPLX_CODELLAMA = "codellama-70b-instruct"
    PPLX_MISTRAL = "mistral-7b-instruct"
    PPLX_MIXTRAL = "mixtral-8x7b-instruct"
    PPLX_7B_ONLINE = "pplx-7b-online"
    PPLX_70B_ONLINE = "pplx-70b-online"

    ANTHROPIC_CALUDE_3_OPUS = "claude-3-opus-20240229"
    ANTHROPIC_CALUDE_3_SONNET = "claude-3-sonnet-20240229"
    ANTHROPIC_CALUDE_3_HAIKU = "claude-3-haiku-20240307"
    ANTHROPIC_CALUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"

    MODEL_MAP = {
        "openai": {
            "gpt3": GPT_3_OPENAI,
            "gpt4": GPT_4_OPENAI,
        },
        "azure": {
            "gpt3": GPT_3_AZURE,
            "gpt4": GPT_4_AZURE,
        },
        "bedrock": {
            "claude1": CLAUDE1,
            "claude2": CLAUDE2,
            "claude2-1": CLAUDE21,
            "claude3-sonnet": CLAUDE3_SONNET,
            "claude3-haiku": CLAUDE3_HAIKU,
            "titan-express": TITAN_EXPRESS,
            "titan-lite": TITAN_LITE,
            "jurassic-ultra": JURASSIC_ULTRA,
        },
        "pplx": {
            "sonar-small-chat": PPLX_SMALL_CHAT,
            "sonar-small-online": PPLX_SMALL_ONLINE,
            "sonar-medium-chat": PPLX_MIDIUM_CHAT,
            "sonar-medium-online": PPLX_MIDIUM_ONLINE,
            "codellama": PPLX_CODELLAMA,
            "mistral": PPLX_MISTRAL,
            "mixtral": PPLX_MIXTRAL,
            "pplx-7b-online": PPLX_7B_ONLINE,
            "pplx-70b-online": PPLX_70B_ONLINE,
        },
        "anthropic": {
            "claude-3-haiku": ANTHROPIC_CALUDE_3_HAIKU,
            "claude-3-sonnet": ANTHROPIC_CALUDE_3_SONNET,
            "claude-3-opus": ANTHROPIC_CALUDE_3_OPUS,
            "claude-3-5-sonnet": ANTHROPIC_CALUDE_3_5_SONNET,
        },
    }

    def __init__(
        self,
        prompt_file: str = None,
        base: Literal["openai", "azure", "bedrock", "pplx", "anthropic"] = "azure",
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
            "titan-express",
            "titan-lite",
            "jurassic-ultra",
            "sonar-small-chat",
            "sonar-small-online",
            "sonar-medium-chat",
            "sonar-medium-online",
            "pplx-7b-online",
            "pplx-70b-online" "claude-3-haiku",
            "claude-3-sonnet",
            "claude-3-opus",
            "claude-3-5-sonnet",
        ] = "gpt4",
    ) -> None:
        self.base = base
        self.use_model = use_model
        self.model = self.choose_model()
        if prompt_file:
            self.prompt = open(prompt_file).read()

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def all_models(self):
        return [model for models in self.MODEL_MAP.values() for model in models.keys()]

    def choose_model(self):

        if self.use_model not in self.all_models():
            print("NOT AVAILABLE MODEL", self.MODEL_MAP.keys())
            return

        if self.base == "openai":
            return OpenAI(self.MODEL_MAP[self.base][self.use_model])
        elif self.base == "azure":
            return AzureOpenAI(self.MODEL_MAP[self.base][self.use_model])
        elif self.base == "bedrock":
            return BEDROCK(self.MODEL_MAP[self.base][self.use_model])
        elif self.base == "pplx":
            return PPLX_AI(self.MODEL_MAP[self.base][self.use_model])
        elif self.base == "anthropic":
            return Anthropic(self.MODEL_MAP[self.base][self.use_model])

    def get_response(
        self,
        temperature=0,
        max_tokens=256,
        top_p=1,
    ) -> str:

        response = self.model.get_response(
            query=self.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        print("======", self.base, self.use_model, "RESULT", "======")
        print(response)
        print("======", "\n")
        response: str = re.findall(r"[\{\[].*[\}\]]", response, re.DOTALL)[0]
        response: str = re.sub(r",\n\s*\}", "\n}", response)
        response: str = re.sub(r"\\", "", response)
        return response


def test():
    # llms = LLM("./test.txt")

    client = LLM(
        prompt_file="./prompt/enhance_vectordb_query.txt",
        base="azure",
        use_model="gpt4",
    )
    client.get_response(max_tokens=3000)


if __name__ == "__main__":
    test()
    pass

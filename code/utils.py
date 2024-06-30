import csv
from functools import wraps
import os
import json
from time import time
from typing import Any, Callable, Literal
from pathlib import Path
import sys
import tiktoken
from joblib import Parallel, delayed

sys.path.append("../vector_db")
from qdrant.lib.llm import LLM


def save_pair(
    config: dict,
    purpose: str,
    prompt: str,
    response: str,
):
    if purpose not in config["prompt-response-pairs"]:
        config["prompt-response-pairs"][purpose] = []
    config["prompt-response-pairs"][purpose].append({"prompt": prompt, "response": response})


def update_token_usage(
    config: dict,
    purpose: str,
    input: int,
    output: int,
):
    if purpose in config["token_usage"]:
        config["token_usage"][purpose]["in"] += input
        config["token_usage"][purpose]["out"] += output
    else:
        config["token_usage"][purpose] = {"in": input, "out": output}


def get_llm(
    config: dict,
    purpose: Literal["default", "translation"],
) -> LLM:
    """retrieves LLM instance from vector_db/qdrant/lib/llm library

    Args:
        config (dict): configuration dict
        purpose : which llm configuration to use. see config file.

    Raises:
        NotImplementedError: if purpose was not defined in the config file

    Returns:
        LLM: LLM instance from vector_db library
    """
    if config["use_model"][purpose] == "opus":
        llm = LLM(base="anthropic", use_model="claude-3-opus")
    elif config["use_model"][purpose] == "sonnet":
        llm = LLM(base="anthropic", use_model="claude-3-5-sonnet")
    elif config["use_model"][purpose] == "gpt3" or config["use_model"][purpose] == "gpt4":
        llm = LLM(use_model=config["use_model"][purpose])
    elif config["use_model"][purpose] == "gpt4o":
        print("loading gpt4o...")
        llm = LLM(base="openai", use_model="gpt4o")
    else:
        raise NotImplementedError

    return llm


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_prompt_from_file(filename: str) -> str:
    """retrieve str from prompt file

    Args:
        filename (str): name of prompt file (e.g. example_prompt.txt)

    Returns:
        str: python str of prompt
    """
    with open(os.path.join("../prompt", filename), "r") as f:
        prompt = f.read()
    return prompt


def save_to_json(config: dict):
    """saves python dictionary data to json

    Args:
        data (dict): data to save
    """

    file_path = Path("../output", config["mode"], "json", config["use_model"]["default"], config["theme"] + ".json")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w+") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Log saved to: {file_path}")


def save_result_to_csv_market_function(theme: str, result: dict, filename: str):
    if len(result) == 0:
        print(f"result dict length 0 for {filename}. could not save csv...")
        return

    fieldnames = [
        "theme",
        "function",
        "function_description",
        "market",
        "usecase_title",
        "market_resolution",
        "market_problem",
        "usecase_description",
        "technology_function",
        "difficulty",
        "evidence_name",
        "evidence_description",
    ]

    def create_row(
        theme: str = "",
        function: str = "",
        function_description: str = "",
        market: str = "",
        usecase_title: str = "",
        market_resolution: str = "",
        market_problem: str = "",
        usecase_description: str = "",
        technology_function: list[str] = [""],
        difficulty: str = "",
        evidence_name: str = "",
        evidence_description: str = "",
    ):
        row_dict = {
            "theme": theme,
            "market": market,
            "function": function,
            "function_description": function_description,
            "usecase_title": usecase_title,
            "market_resolution": market_resolution,
            "market_problem": market_problem,
            "usecase_description": usecase_description,
            "technology_function": "\n".join(technology_function),
            "difficulty": difficulty,
            "evidence_name": evidence_name,
            "evidence_description": evidence_description,
        }
        return row_dict

    file_path = Path("../output/market-function/csv", filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(create_row(theme=theme))

        for item in result["technology_functions"]:
            writer.writerow(create_row(function=item["name"], function_description=item["description"]))

        for per_market_res in result["usecase"]:
            writer.writerow(create_row(market=per_market_res["market"]))
            for usecase in per_market_res["usecase"]:
                writer.writerow(
                    create_row(
                        usecase_title=usecase["title"],
                        technology_function=usecase["technology_function"],
                        market_resolution=usecase["market_resolution"],
                        market_problem=usecase["market_problem"],
                        usecase_description=usecase["description"],
                        difficulty=usecase["difficulty"],
                    )
                )
                for evidence in usecase["evidence"]:
                    writer.writerow(
                        create_row(
                            evidence_name=evidence["name"],
                            evidence_description=evidence["description"],
                        )
                    )
        print(f"csv file logged to {file_path}")


def translate_market_function(config):
    result_en = config["result_en"]
    result_jp = {}

    result_jp["technology_functions"] = translate(config, result_en["technology_functions"])
    with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
        result_jp["usecase"] = parallel(delayed(translate)(config, partial_res) for partial_res in result_en["usecase"])

    config["result_jp"] = result_jp


def save_result_to_csv_market_only(theme: str, result: dict, filename: str):
    if len(result) == 0:
        print(f"result dict length 0 for {filename}. could not save csv...")
        return

    fieldnames = [
        "theme",
        "market",
        "usecase_title",
        "market_resolution",
        "market_problem",
        "usecase_description",
        "difficulty",
        "evidence_name",
        "evidence_description",
    ]

    def create_row(
        theme: str = "",
        market: str = "",
        usecase_title: str = "",
        market_resolution: str = "",
        market_problem: str = "",
        usecase_description: str = "",
        difficulty: str = "",
        evidence_name: str = "",
        evidence_description: str = "",
    ):
        row_dict = {
            "theme": theme,
            "market": market,
            "usecase_title": usecase_title,
            "market_resolution": market_resolution,
            "market_problem": market_problem,
            "usecase_description": usecase_description,
            "difficulty": difficulty,
            "evidence_name": evidence_name,
            "evidence_description": evidence_description,
        }
        return row_dict

    file_path = Path("../output/market-only/csv", filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(create_row(theme=theme))

        for per_market_res in result["usecase"]:
            writer.writerow(create_row(market=per_market_res["market"]))
            for usecase in per_market_res["usecase"]:
                writer.writerow(
                    create_row(
                        usecase_title=usecase["title"],
                        market_resolution=usecase["market_resolution"],
                        market_problem=usecase["market_problem"],
                        usecase_description=usecase["description"],
                        difficulty=usecase["difficulty"],
                    )
                )
                for evidence in usecase["evidence"]:
                    writer.writerow(
                        create_row(
                            evidence_name=evidence["name"],
                            evidence_description=evidence["description"],
                        )
                    )
        print(f"csv file logged to {file_path}")


def translate_market_only(config):
    result_en = config["result_en"]
    result_jp = {}

    with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
        result_jp["usecase"] = parallel(delayed(translate)(config, partial_res) for partial_res in result_en["usecase"])

    config["result_jp"] = result_jp


def save_result_to_csv_technology_only(theme: str, result: dict, filename: str):
    if len(result) == 0:
        print(f"result dict length 0 for {filename}. could not save csv...")
        return

    fieldnames = [
        "theme",
        "purpose",
        "category",
        "layer1",
        "l1_desc",
        "layer2",
        "l2_desc",
        "layer3",
        "l3_desc",
    ]

    def create_row(
        theme: str = "",
        purpose: str = "",
        category: str = "",
        layer1: str = "",
        l1_desc: str = "",
        layer2: str = "",
        l2_desc: str = "",
        layer3: str = "",
        l3_desc: str = "",
    ):
        row_dict = {
            "theme": theme,
            "purpose": purpose,
            "category": category,
            "layer1": layer1,
            "l1_desc": l1_desc,
            "layer2": layer2,
            "l2_desc": l2_desc,
            "layer3": layer3,
            "l3_desc": l3_desc,
        }
        return row_dict

    file_path = Path("../output/technology-only/csv", filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(create_row(theme=theme))

        # principle
        writer.writerow(create_row(purpose="Principle"))
        for item in result["principle"]:
            writer.writerow(create_row(layer1=item["title"], l1_desc=item["description"]))

        writer.writerow(create_row(purpose="Mechanism"))
        for step_info in result["mechanism"]:
            writer.writerow(
                create_row(
                    category=step_info["step"],
                    layer1=step_info["title"],
                    l1_desc=step_info["description"],
                )
            )
            for l2_item in step_info["layer2"]:
                writer.writerow(
                    create_row(
                        layer2=l2_item["title_layer2"],
                        l2_desc=l2_item["description"],
                    )
                )
                for l3_item in l2_item["example"]:
                    writer.writerow(
                        create_row(
                            layer3=l3_item["title_layer3"],
                            l3_desc=l3_item["description"],
                        )
                    )

        # requirement
        writer.writerow(create_row(purpose="Requirement"))
        for category, value_list in result["requirement"].items():
            writer.writerow(create_row(category=category))
            for l1_item in value_list:
                writer.writerow(create_row(layer1=l1_item["title_layer1"], l1_desc=l1_item["description"]))
                for l2_item in l1_item["implementation"]:
                    writer.writerow(
                        create_row(
                            layer2=l2_item["title_layer2"],
                            l2_desc=l2_item["description"],
                        )
                    )
                    for l3_item in l2_item["example"]:
                        writer.writerow(
                            create_row(
                                layer3=l3_item["title_layer3"],
                                l3_desc=l3_item["description"],
                            )
                        )
        print(f"csv file logged to {file_path}")


def translate_technology_only(config):
    # principle
    config["result_jp"]["principle"] = translate(config, input=config["result_en"]["principle"])

    # mechanism
    with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
        results = parallel(delayed(translate)(config, step_info) for step_info in config["result_en"]["mechanism"])
    results = sorted(results, key=lambda x: x["step"])
    config["result_jp"]["mechanism"] = results

    # requirement
    config["result_jp"]["requirement"] = {}
    for req_group, req_analysis_list in config["result_en"]["requirement"].items():
        with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
            results = parallel(delayed(translate)(config, analysis) for analysis in req_analysis_list)
        config["result_jp"]["requirement"][req_group] = results


def translate(config: dict, input: str) -> dict:
    """translates english results to japanese.
    process per market

    Args:
        config (dict): configuration dict
        input (str): input english result to translate. result of 1 market

    Returns:
        dict: {market:"", usecase:[title:......]} output format same as generate_usecase_from_one_market()
    """
    llm = get_llm(config, "translation")

    prompt = read_prompt_from_file("translate_to_japanese.txt")
    prompt = prompt.replace("<input>", json.dumps(input))
    llm.set_prompt(prompt)
    jap_result = llm.get_response(max_tokens=4096)

    # token counting
    update_token_usage(config, "translation", llm.model.usage_in, llm.model.usage_out)
    save_pair(config, "translation", prompt=prompt, response=jap_result)
    return json.loads(jap_result)


def calculate_time(fn: Callable):
    """decorator for calculating time a function takes to execute

    IMPORTANT: "elapsed_time" will be appended to the end of original function's return values

    Args:
        fn (Callable): function to wrap decorator around
    """

    @wraps(fn)
    def new_fn(*args, **kwargs) -> tuple[Any, float]:
        s_time = time()
        res = fn(*args, **kwargs)
        elapsed_time = time() - s_time
        print(f"\n----- {fn.__qualname__.split('.')[0]}: {elapsed_time} seconds -----\n")
        return res, elapsed_time

    return new_fn


if __name__ == "__main__":
    with open("../output/technology-only/json/sonnet/AR.json") as f:
        res = json.load(f)

    # translate_market_function(res)
    # save_to_json(res)
    save_result_to_csv_technology_only(res["theme"], res["result_en"], "test.csv")

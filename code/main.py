import json
from joblib import Parallel, delayed
from utils import (
    save_to_json,
    save_result_to_csv_market_function,
    translate_market_function,
    translate_market_only,
    save_result_to_csv_market_only,
    translate_technology_only,
    save_result_to_csv_technology_only,
)
from time import time
from typing import Literal
from argparse import ArgumentParser

from market_market_generation import Market_Market_Generation
from market_per_market_analysis import Market_Per_Market_Analysis
from technology_function import Technology_Function
from technology_mechanism import Technology_Mechanism
from technology_principle import Technology_Principle
from technology_requirement import Technology_Requirement


def market_function(config):
    begin = time()
    ##### 1. Generate Technology-Function and Markets
    functions = Technology_Function(config)()
    markets = Market_Market_Generation(config)()

    ##### 2. Generate Market-Resolution-Problem and Usecase with Technology Function
    per_market_analysis_list = [
        Market_Per_Market_Analysis(config, target_market, functions) for target_market in markets["markets"]
    ]
    with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
        parallel(delayed(market)() for market in per_market_analysis_list)

    ##### 3. Process English Results
    config["result_en"]["technology_functions"] = config["result_raw"]["technology_function"]["result"]["functions"]
    config["result_en"]["usecase"] = [usecase["result"] for usecase in config["result_raw"]["usecase"]]

    ##### 4. Translate
    if config["translate"]:
        translate_market_function(config)

    ##### Save Time
    elapsed_time = time() - begin
    config["total_time"] = elapsed_time
    print(f"\n##### {config['mode']}: {elapsed_time} seconds #####\n")

    ##### 5. Log Output
    save_to_json(config)
    save_result_to_csv_market_function(
        theme=config["theme"],
        result=config["result_en"],
        filename=config["use_model"]["default"] + "/" + config["theme"] + "_en.csv",
    )
    if config["translate"]:
        save_result_to_csv_market_function(
            theme=config["theme"],
            result=config["result_jp"],
            filename=config["use_model"]["default"] + "/" + config["theme"] + "_jp.csv",
        )


def market_only(config):
    begin = time()
    ##### 1. Generate Market
    market_generation_model = Market_Market_Generation(config)
    markets = market_generation_model()

    ##### 2. Generate Market-Resolution-Problem and Usecase per market
    per_market_analysis_list = [
        Market_Per_Market_Analysis(config, target_market, None) for target_market in markets["markets"]
    ]
    with Parallel(n_jobs=config["parallel_jobs"], backend="threading") as parallel:
        parallel(delayed(market)() for market in per_market_analysis_list)

    ##### 3. Process English Results
    config["result_en"]["usecase"] = [usecase["result"] for usecase in config["result_raw"]["usecase"]]

    ##### 4. Translate
    if config["translate"]:
        translate_market_only(config)

    ##### Save Time
    elapsed_time = time() - begin
    config["total_time"] = elapsed_time
    print(f"\n##### {config['mode']}: {elapsed_time} seconds #####\n")

    ##### 5. Log Output
    save_to_json(config)
    save_result_to_csv_market_only(
        theme=config["theme"],
        result=config["result_en"],
        filename=config["use_model"]["default"] + "/" + config["theme"] + "_en.csv",
    )
    if config["translate"]:
        save_result_to_csv_market_only(
            theme=config["theme"],
            result=config["result_jp"],
            filename=config["use_model"]["default"] + "/" + config["theme"] + "_jp.csv",
        )


def technology_only(config):
    begin = time()
    ##### 1. Get Results
    Technology_Principle(config)()
    Technology_Mechanism(config)()
    Technology_Requirement(config)()

    ##### 2. Process English Results from raw results
    # principle
    config["result_en"]["principle"] = config["result_raw"]["technology_principle"]["result"]
    # requirement
    config["result_en"]["requirement"] = config["result_raw"]["technology_requirement_layer1"]["result"]
    for res in config["result_raw"]["technology_requirement_layer23"]["result"]:
        for item in config["result_en"]["requirement"][res["category"]]:
            if item["title_layer1"] != res["subcategory"]:
                continue
            item["implementation"] = res["result"]
    # mechanism
    config["result_en"]["mechanism"] = config["result_raw"]["technology_mechanism_layer1"]["result"]
    for res in config["result_raw"]["technology_mechanism_layer23"]["result"]:
        for step_info in config["result_en"]["mechanism"]:
            if str(step_info["step"]) != str(res["step"]):
                continue
            step_info["layer2"] = res["result"]

    ##### 3. Translate
    if config["translate"]:
        translate_technology_only(config)

    ##### Save Time
    elapsed_time = time() - begin
    config["total_time"] = elapsed_time
    print(f"\n##### {config['mode']}: {elapsed_time} seconds #####\n")

    ##### 4. Log Output
    save_to_json(config)
    save_result_to_csv_technology_only(
        theme=config["theme"],
        result=config["result_en"],
        filename=config["use_model"]["default"] + "/" + config["theme"] + "_en.csv",
    )
    if config["translate"]:
        save_result_to_csv_technology_only(
            theme=config["theme"],
            result=config["result_jp"],
            filename=config["use_model"]["default"] + "/" + config["theme"] + "_jp.csv",
        )


def parse_arg(config):
    parser = ArgumentParser()

    parser.add_argument("-T", "--theme", type=str, default="None")
    # parser.add_argument("-T", "--theme", type=str, default="AR")

    args = parser.parse_args()

    if args.theme == "None":
        raise NotImplementedError("specify theme")
    config["theme"] = args.theme


PossibleModes = Literal["market-function", "market-only", "technology-only"]
########## CHANGE ACCORDINGLY ##########
MODE: PossibleModes = "market-function"
# MODE: PossibleModes = "market-only"
# MODE: PossibleModes = "technology-only"
########################################


def main():
    with open("_config.json", "r") as f:
        config: dict = json.load(f)
    config["mode"] = MODE
    parse_arg(config)

    match MODE:
        case "market-function":
            market_function(config)
        case "market-only":
            market_only(config)
        case "technology-only":
            technology_only(config)
        case _:
            raise NotImplementedError("mode not implemented")


if __name__ == "__main__":
    main()

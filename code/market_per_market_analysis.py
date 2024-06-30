import json
from dotenv import load_dotenv
from joblib import Parallel, delayed
from utils import (
    read_prompt_from_file,
    get_llm,
    calculate_time,
    update_token_usage,
    save_pair,
)


class Market_Per_Market_Analysis:
    def __init__(self, config, target_market, functions):
        self.config = config
        self.llm = get_llm(config, "default")

        self.target_market = target_market
        if functions == None:
            self.only_market = True
        else:
            self.only_market = False
            self.technology_functions = functions

        self.prompt_filename = "market_market.txt"

    def get_selfprompt(self, filename, target_market):
        # set prompt
        selfprompt = read_prompt_from_file(filename)
        selfprompt = selfprompt.replace("<market>", json.dumps(target_market))
        self.llm.set_prompt(selfprompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)
        update_token_usage(
            self.config, "market_resolution_and_problem", self.llm.model.usage_in, self.llm.model.usage_out
        )
        save_pair(self.config, "market_resolution_and_problem_selfprompt", selfprompt, result)
        result = json.loads(result)
        return result["prompt"]

    @calculate_time
    def generate_market_resolution_and_problem(self, target_market):
        # get self prompt
        prompt = self.get_selfprompt(
            "market_resolution_problem_per_market_selfprompt.txt",
            target_market,
        )
        # set prompt
        prompt += read_prompt_from_file("market_resolution_problem_per_market.txt")
        self.llm.set_prompt(prompt)
        result = self.llm.get_response(max_tokens=4096)
        return prompt, result

    @calculate_time
    def generate_usecase(self, market_problems):
        if self.only_market:
            prompt = read_prompt_from_file("market_usecase_per_market.txt")
            prompt = prompt.replace("<technology>", self.config["theme"])
            prompt = prompt.replace("<market_information>", json.dumps(market_problems))
        else:
            prompt = read_prompt_from_file("market_usecase_per_market_with_function.txt")
            prompt = prompt.replace("<technology>", self.config["theme"])
            prompt = prompt.replace("<technology_function>", json.dumps(self.technology_functions))
            prompt = prompt.replace("<market_information>", json.dumps(market_problems))

        self.llm.set_prompt(prompt)
        result = self.llm.get_response(max_tokens=4096)
        return prompt, result

    def process_result(self, market_problems, time1, usecase, time2):
        # res = {
        #     "time": time1 + time2,
        #     "result": "",
        # }
        # if "per_market_analysis" not in self.config["result_raw"]:
        #     self.config["result_raw"]["per_market_analysis"] = []
        # self.config["result_raw"]["per_market_analysis"].append(res)

        if "market_resolution_problem" not in self.config["result_raw"]:
            self.config["result_raw"]["market_resolution_problem"] = []
        self.config["result_raw"]["market_resolution_problem"].append({"time": time1, "result": market_problems})

        if "usecase" not in self.config["result_raw"]:
            self.config["result_raw"]["usecase"] = []
        self.config["result_raw"]["usecase"].append({"time": time2, "result": usecase})

    def __call__(self):
        # Market resolution and problem
        (prompt, market_problems), time1 = self.generate_market_resolution_and_problem(self.target_market)
        update_token_usage(
            self.config, "market_resolution_and_problem", self.llm.model.usage_in, self.llm.model.usage_out
        )
        save_pair(self.config, "market_resolution_and_problem", prompt, market_problems)
        market_problems = json.loads(market_problems)

        # Usecase generation from market problem and technology function
        self.llm = get_llm(self.config, "default")
        (prompt, usecase), time2 = self.generate_usecase(market_problems)
        update_token_usage(self.config, "market_usecase", self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, "market_usecase", prompt, usecase)
        usecase = json.loads(usecase)

        self.process_result(market_problems, time1, usecase, time2)
        return usecase


if __name__ == "__main__":
    load_dotenv()

    with open("_config.json", "r") as f:
        config: dict = json.load(f)
    # config["theme"] = "AR"
    model = Market_Per_Market_Analysis(config)
    model()

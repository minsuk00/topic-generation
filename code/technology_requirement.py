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


class Technology_Requirement:
    def __init__(self, config):
        self.config = config
        self.name = "technology_requirement"
        self.llm = get_llm(config, "default")

    def get_selfprompt(self):
        # set prompt
        selfprompt = read_prompt_from_file("technology_requirement_selfprompt.txt")
        selfprompt = selfprompt.replace("<technology>", self.config["theme"])
        self.llm.set_prompt(selfprompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)
        # update_token_usage(self.config, selfprompt, result, "requirement")
        update_token_usage(
            self.config,
            self.name,
            self.llm.model.usage_in,
            self.llm.model.usage_out,
        )
        save_pair(self.config, self.name + "_selfprompt", selfprompt, result)
        result = json.loads(result)
        return result["prompt"]

    @calculate_time
    def get_layer1_output(self):
        # get self prompt
        prompt = self.get_selfprompt()
        # set prompt
        prompt += read_prompt_from_file("technology_requirement_L1.txt")
        self.llm.set_prompt(prompt)
        result = self.llm.get_response(max_tokens=4096)
        # update_token_usage(self.config, prompt, result, "requirement")
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name + "_layer1", prompt, result)
        return json.loads(result)

    @calculate_time
    def get_layer23_output(self, l1_data: dict):
        subcat_list = []
        for category, subcategory_list in l1_data.items():
            for subcat in subcategory_list:
                subcat_list.append((category, subcat["title_layer1"], subcat["description"]))

        with Parallel(n_jobs=self.config["parallel_jobs"], backend="threading") as parallel:
            results = parallel(
                delayed(self.analyze_layer23_individually)(cat, subcat, desc) for cat, subcat, desc in subcat_list
            )
        return results

    def analyze_layer23_individually(self, cat: str, subcat: str, desc: str):
        prompt = read_prompt_from_file("technology_requirement_L23.txt")
        prompt = prompt.replace("<technology>", self.config["theme"])
        prompt = prompt.replace("<category>", cat)
        prompt = prompt.replace("<subcategory>", subcat)
        prompt = prompt.replace("<description>", desc)
        self.llm.set_prompt(prompt)

        result = self.llm.get_response(max_tokens=4096)
        # update_token_usage(self.config, prompt, result, "requirement")
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name + "_layer23", prompt, result)
        return json.loads(result)

    def process_result(self, l1_result, time1, l23_result, time2):
        self.config["result_raw"][self.name + "_layer1"] = {}
        self.config["result_raw"][self.name + "_layer1"]["time"] = time1
        self.config["result_raw"][self.name + "_layer1"]["result"] = l1_result

        self.config["result_raw"][self.name + "_layer23"] = {}
        self.config["result_raw"][self.name + "_layer23"]["time"] = time2
        self.config["result_raw"][self.name + "_layer23"]["result"] = l23_result

    def __call__(self):
        l1_result, time1 = self.get_layer1_output()
        l23_result, time2 = self.get_layer23_output(l1_result)
        self.process_result(l1_result, time1, l23_result, time2)


if __name__ == "__main__":
    load_dotenv()

    with open("technology_template.json", "r") as f:
        config: dict = json.load(f)
    config["technology"] = "AR"
    model = Technology_Requirement(config)
    model()

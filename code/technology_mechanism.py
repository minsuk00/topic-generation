import json
from joblib import Parallel, delayed
from dotenv import load_dotenv
from utils import (
    read_prompt_from_file,
    get_llm,
    calculate_time,
    update_token_usage,
    save_pair,
)


class Technology_Mechanism:
    def __init__(self, config):
        self.config = config
        self.name = "technology_mechanism"
        self.llm = get_llm(config, "default")

    def get_selfprompt(self):
        # set prompt
        selfprompt = read_prompt_from_file("technology_mechanism_selfprompt.txt")
        selfprompt = selfprompt.replace("<technology>", self.config["theme"])
        self.llm.set_prompt(selfprompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name + "_selfprompt", selfprompt, result)
        result = json.loads(result)
        return result["prompt"]

    @calculate_time
    def get_layer1_output(self):
        prompt = self.get_selfprompt()
        # set prompt
        prompt += read_prompt_from_file("technology_mechanism_L1.txt")
        self.llm.set_prompt(prompt)
        result = self.llm.get_response(max_tokens=4096)
        # update_token_usage(self.config, prompt, result, "mechanism")
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name + "_layer1", prompt, result)
        return json.loads(result)

    @calculate_time
    def get_layer23_output(self, l1_data: dict):
        total_step_num = len(l1_data)
        with Parallel(n_jobs=self.config["parallel_jobs"], backend="threading") as parallel:
            results = parallel(
                delayed(self.analyze_layer23_individually)(step_info, total_step_num) for step_info in l1_data
            )
        return results

    def analyze_layer23_individually(self, step_info: dict, total_step_num: int):
        prompt = read_prompt_from_file("technology_mechanism_L23.txt")
        prompt = prompt.replace("<technology>", self.config["theme"])
        prompt = prompt.replace("<current_step>", str(step_info["step"]))
        prompt = prompt.replace("<total_steps>", str(total_step_num))
        prompt = prompt.replace("<title>", step_info["title"])
        prompt = prompt.replace("<description>", step_info["description"])
        self.llm.set_prompt(prompt)

        result = self.llm.get_response(max_tokens=4096)
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
    model = Technology_Mechanism(config)
    model()

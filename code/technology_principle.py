import json
from dotenv import load_dotenv
from utils import (
    read_prompt_from_file,
    get_llm,
    calculate_time,
    update_token_usage,
    save_pair,
)


class Technology_Principle:
    def __init__(self, config):
        self.config = config
        self.name = "technology_principle"
        self.llm = get_llm(config, "default")

    def get_selfprompt(self):
        # set prompt
        selfprompt = read_prompt_from_file("technology_principle_selfprompt.txt")
        selfprompt = selfprompt.replace("<technology>", self.config["theme"])
        self.llm.set_prompt(selfprompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)
        # update_token_usage(self.config, selfprompt, result, "principle")
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name + "_selfprompt", selfprompt, result)
        result = json.loads(result)
        return result["prompt"]

    def process_result(self, result, time):
        self.config["result_raw"][self.name] = {}
        self.config["result_raw"][self.name]["time"] = time
        self.config["result_raw"][self.name]["result"] = result

    @calculate_time
    def forward(self):
        # set prompt
        prompt = self.get_selfprompt()
        prompt += read_prompt_from_file("technology_principle.txt")
        self.llm.set_prompt(prompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)

        return prompt, result

    def __call__(self):
        (prompt, result), time = self.forward()
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        save_pair(self.config, self.name, prompt, result)
        result = json.loads(result)
        self.process_result(result, time)


if __name__ == "__main__":
    load_dotenv()

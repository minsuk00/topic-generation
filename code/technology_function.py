import json
from dotenv import load_dotenv

from utils import (
    read_prompt_from_file,
    get_llm,
    calculate_time,
    update_token_usage,
    save_pair,
)


class Technology_Function:
    def __init__(self, config):
        self.config = config
        self.name = "technology_function"
        self.llm = get_llm(config, "default")
        self.prompt_filename = "technology_function.txt"

    def process_result(self, result, time):
        # self.config["result_en"][self.name] = result
        # self.config["result_raw"][self.name] = result
        self.config["result_raw"][self.name] = {}
        self.config["result_raw"][self.name]["time"] = time
        self.config["result_raw"][self.name]["result"] = result

    @calculate_time
    def forward(self):
        # set prompt
        prompt = read_prompt_from_file(self.prompt_filename)
        prompt = prompt.replace("<technology>", self.config["theme"])
        self.llm.set_prompt(prompt)
        # call llm
        result = self.llm.get_response(max_tokens=4096)
        # result = json.loads(result)
        return prompt, result

    def __call__(self):
        (prompt, result), time = self.forward()
        # update token usage
        update_token_usage(self.config, self.name, self.llm.model.usage_in, self.llm.model.usage_out)
        # save prompt-response pair
        save_pair(self.config, self.name, prompt, result)
        # process result
        result = json.loads(result)
        self.process_result(result, time)
        return result


if __name__ == "__main__":
    load_dotenv()

    with open("_config.json", "r") as f:
        config: dict = json.load(f)
    config["theme"] = "AR"
    model = Technology_Function(config)
    model()

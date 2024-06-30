import dotenv

from llm import LLM
import json
from pprint import pprint

from tqdm import tqdm
from generate_description import GENERATE_TECHNOLOGY_DESCRIPTION, TECHNOLOGY_DESCRIPTION
from fact_check import FACT_CHECK
from typing import Literal
import os

dotenv.load_dotenv()


class ENHANCE_QUERY:
    def __init__(
        self,
        keyword: str,
        japanese_keyword: str,
        description: str = "",
        japanese_description: str = "",
    ):
        self.keyword = keyword
        self.japanese_keyword = japanese_keyword
        self.description = description
        self.japanese_description = japanese_description

    def get_dict(self):
        """
        Returns:
            {
                title: Needs Title
                japanese_title: 日本語Needs名
                description: Needs Description
                japanese_description: 日本語Needs概要
            }
        """
        return {
            "keyword": self.keyword,
            "japanese_keyword": self.japanese_keyword,
            "description": self.description,
            "japanese_description": self.japanese_description,
        }


class ALL_ENHANCE_QUERY:
    def __init__(self, enhance_query_list: list):
        self.enhance_query_list = enhance_query_list
        self.data = list[ENHANCE_QUERY](self.read_enhance_query_list())

    def read_enhance_query_list(self):
        return [
            ENHANCE_QUERY(
                keyword=query["keyword"],
                japanese_keyword=query["japanese_keyword"],
                description=query["description"],
                japanese_description=query["japanese_description"],
            )
            for query in self.enhance_query_list
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]

    def get_keywords(self):
        return [item.keyword for item in self.data]


class GENERATE_ENHANCE_QUERY(LLM):
    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "/generate_enhance_query.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_query"

    N_QUERY_SPLIT = 20

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
        ] = "gpt3",
        base: Literal[
            "openai",
            "azure",
            "bedrock",
        ] = "azure",
        temperature=0,
        max_tokens=3000,
        top_p=0,
    ) -> None:
        self.data = ALL_ENHANCE_QUERY([])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
            base=base,
        )

    def save_data(self):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{self.technology}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.data.get_data(), f, ensure_ascii=False)
            f.close()

    def split_number_with_interval(self, total: int):
        return [self.N_QUERY_SPLIT] * (total // self.N_QUERY_SPLIT) + [
            total % self.N_QUERY_SPLIT
        ]

    def generate_enhance_query_once(self, number_of_query: int):
        keywords = "\n".join(self.data.get_keywords())

        self.prompt = self.prompt.replace("<n_query>", str(number_of_query))
        self.prompt = self.prompt.replace("<keywords>", keywords)

        result = ALL_ENHANCE_QUERY(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )["result"]
        )
        self.data.data.extend(result.data)

    def generate_enhance_query(self):
        self.prompt = self.prompt.replace("<technology>", self.technology)
        self.prompt = self.prompt.replace("<query>", self.query)
        for number_of_query in tqdm(
            self.split_number_with_interval(self.number_of_query), desc="GENERATE QUERY"
        ):
            prompt = self.prompt
            self.generate_enhance_query_once(number_of_query=number_of_query)
            self.save_data()
            self.prompt = prompt

    def generate(self, technology: str, query: str, number_of_query: int):
        self.technology = technology
        self.query = query
        self.number_of_query = number_of_query
        self.generate_enhance_query()


class GENERAL_CONCEPT:
    def __init__(
        self,
        concept: str,
        japanese_concept: str,
        description: str = "",
        japanese_description: str = "",
    ):
        self.concept = concept
        self.japanese_concept = japanese_concept
        self.description = description
        self.japanese_description = japanese_description

    def get_dict(self):
        """
        Returns:
            {
                title: Concept Name
                japanese_concept: 日本語Concept名
                description: Concept Description
                japanese_description: 日本語Concept概要
            }
        """
        return {
            "concept": self.concept,
            "japanese_concept": self.japanese_concept,
            "description": self.description,
            "japanese_description": self.japanese_description,
        }


class ALL_GENERAL_CONCEPT:
    def __init__(self, general_concept_list: list):
        self.general_concept_list = general_concept_list
        self.data = list[GENERAL_CONCEPT](self.read_general_concept_list())

    def read_general_concept_list(self):
        return [
            GENERAL_CONCEPT(
                concept=concept["concept"],
                japanese_concept=concept["japanese_concept"],
                description=concept["description"],
                japanese_description=concept["japanese_description"],
            )
            for concept in self.general_concept_list
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]

    def get_concepts(self):
        return [item.concept for item in self.data]


class GENERATE_GENERAL_TOPIC(LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "network/generate_general_topics.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_general_topic"

    N_SPLIT = 20

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
        ] = "claude1",
        base: Literal[
            "openai",
            "azure",
            "bedrock",
        ] = "azure",
        temperature=0,
        max_tokens=3000,
        top_p=0,
    ) -> None:
        # self.data: dict[str, MARKET_NEEDS] = {}
        self.data = ALL_GENERAL_CONCEPT([])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
            base=base,
        )

    def save_data(self):
        os.makedirs(f"{self.BASE_DIR}/{self.SAVE_DIR}", exist_ok=True)
        with open(
            f"{self.BASE_DIR}/{self.SAVE_DIR}/{self.technology}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.data.get_data(), f, ensure_ascii=False)
            f.close()

    def enhance_description(self, query: str):
        technology_string = f"{self.technology} Query: {query}"
        client = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude2")
        client.generate(technology=technology_string)
        return client.get_dict()

    def split_number_with_interval(self, total: int):
        return [self.N_SPLIT] * (total // self.N_SPLIT) + [total % self.N_SPLIT]

    def research_papers(self, papers: list):
        items = [
            "title",
            "abstract",
            "publication_date",
            "cited_by_count",
            "doi",
            "score",
            "concepts",
        ]
        return [{item: paper[item] for item in items} for paper in papers]

    def research_papers_string(self, papers: list):
        return "\n".join(
            [
                f'{index+1}. {paper["title"]} ({paper["doi"]}), CitationCount: {paper["cited_by_count"]}, Concepts: {[concept["name"] for concept in paper["concepts"] if int(concept["level"]) > 1]}'
                for index, paper in enumerate(papers)
            ]
        )

    def get_research_papers(self, description: dict) -> TECHNOLOGY_DESCRIPTION:
        technology_string = f"{description}"
        client = FACT_CHECK(n_result=20)
        client.check_query(technology_string)
        return client.papers_result

    def generate_general_topics_once(self, n_topics: int):
        concepts = "\n".join(self.data.get_concepts())

        self.prompt = self.prompt.replace("<n_topics>", str(n_topics))
        self.prompt = self.prompt.replace("<concepts>", concepts)
        result = ALL_GENERAL_CONCEPT(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )["result"]
        )

        self.data.data.extend(result.data)

    def generate_general_topics(
        self,
        query: str,
        description: dict,
        papers: list,
    ):
        self.prompt = self.prompt.replace("<technology>", self.technology)
        self.prompt = self.prompt.replace("<query>", query)
        self.prompt = self.prompt.replace(
            "<description>", json.dumps(description, ensure_ascii=False)
        )
        self.prompt = self.prompt.replace(
            "<papers>", self.research_papers_string(papers)
        )

        for n_topics in tqdm(
            self.split_number_with_interval(self.n_topics),
            desc="GENERATE GENERAL TOPIC",
        ):
            prompt = self.prompt
            self.generate_general_topics_once(n_topics=n_topics)
            self.save_data()
            self.prompt = prompt

    def get_general_topics(self, query: str):
        description = self.enhance_description(query=query)
        papers = self.get_research_papers(description=description)
        self.generate_general_topics(
            query=query,
            description=description,
            papers=papers,
        )

    def generate(self, technology: str, query: str, n_topics=10):
        self.technology = technology
        self.n_topics = n_topics
        self.get_general_topics(query=query)


class RELATION:
    def __init__(
        self,
        to: str,
        relation: str,
        japanese_relation: str = "",
    ):
        self.to = to
        self.relation = relation
        self.japanese_relation = japanese_relation

    def get_dict(
        self,
    ):
        return {
            "to": self.to,
            "relation": self.relation,
            "japanese_relation": self.japanese_relation,
        }


class ALL_RELATION:
    def __init__(self, relations: list):
        self.relations = relations
        self.data = list[RELATION](self.read_relations())

    def read_relations(self):
        return [
            RELATION(
                to=items["to"],
                relation=items["relation"],
                japanese_relation=items["japanese_relation"],
            )
            for items in self.relations
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]


class MAIN_CONCEPT:

    def __init__(
        self,
        concept: str,
        japanese_concept: str,
        description: str = "",
        japanese_description: str = "",
        relations: list = [],
        reason: str = "",
    ):
        self.concept = concept
        self.japanese_concept = japanese_concept
        self.description = description
        self.japanese_description = japanese_description
        self.reason = reason
        self.relations = ALL_RELATION(relations)

    def get_dict(self):
        """
        Returns:
            {
                title: Concept Name
                japanese_concept: 日本語Concept名
                description: Concept Description
                japanese_description: 日本語Concept概要
                relation: self.relation,
                japanese_relation: self.japanese_relation,
            }
        """
        return {
            "concept": self.concept,
            "japanese_concept": self.japanese_concept,
            "description": self.description,
            "japanese_description": self.japanese_description,
            "relations": self.relations.get_data(),
            "reason": self.reason,
        }


class ALL_MAIN_CONCEPT:
    def __init__(self, main_concept_list: list):
        self.main_concept_list = main_concept_list
        self.data = list[MAIN_CONCEPT](self.read_main_concept_list())

    def read_main_concept_list(self):
        return [
            MAIN_CONCEPT(
                concept=concept["concept"],
                japanese_concept=concept["japanese_concept"],
                description=concept["description"],
                japanese_description=concept["japanese_description"],
                relations=concept["relations"],
                reason=concept["reason"],
            )
            for concept in self.main_concept_list
        ]

    def get_data(self):
        return [item.get_dict() for item in self.data]

    def get_concepts(self):
        return [item.concept for item in self.data]


class GENERATE_MAIN_TOPIC(LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "network/generate_main_topics.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "generated_main_topic"

    N_SPLIT = 5

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
        ] = "claude1",
        base: Literal[
            "openai",
            "azure",
            "bedrock",
        ] = "azure",
        temperature=0,
        max_tokens=3000,
        top_p=0,
    ) -> None:
        self.data = ALL_MAIN_CONCEPT([])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
            base=base,
        )

    def save_data(self):
        file_dir = f"{self.BASE_DIR}/{self.SAVE_DIR}/{self.technology}"
        os.makedirs(file_dir, exist_ok=True)
        with open(
            f"{file_dir}/{self.general_concept}_{self.use_model}.json",
            "w+",
        ) as f:
            json.dump(self.data.get_data(), f, ensure_ascii=False)
            f.close()

    def enhance_description(self, query: str):
        technology_string = (
            f"{self.technology} (Write more on {self.general_concept}), Query: {query}"
        )
        client = GENERATE_TECHNOLOGY_DESCRIPTION(use_model="claude2")
        client.generate(technology=technology_string)
        return client.get_dict()

    def split_number_with_interval(self, total: int):
        return [self.N_SPLIT] * (total // self.N_SPLIT) + [total % self.N_SPLIT]

    def research_papers(self, papers: list):
        items = [
            "title",
            "abstract",
            "publication_date",
            "cited_by_count",
            "doi",
            "score",
            "concepts",
        ]
        return [{item: paper[item] for item in items} for paper in papers]

    def research_papers_string(self, papers: list):
        return "\n".join(
            [
                f'{index+1}. {paper["title"]} ({paper["doi"]}), CitationCount: {paper["cited_by_count"]}'
                for index, paper in enumerate(papers)
            ]
        )

    def get_research_papers(self, description: dict) -> TECHNOLOGY_DESCRIPTION:
        technology_string = f"{description}"
        client = FACT_CHECK(n_result=20)
        client.check_query(technology_string)
        return client.papers_result

    def generate_general_topics_once(self, n_topics: int):
        concepts = "\n".join(self.data.get_concepts())

        self.prompt = self.prompt.replace("<n_topics>", str(n_topics))
        self.prompt = self.prompt.replace("<concepts>", concepts)
        result = ALL_MAIN_CONCEPT(
            json.loads(
                self.get_response(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            )["result"]
        )
        self.data.data.extend(result.data)

    def generate_main_concepts(
        self,
        query: str,
        description: dict,
        papers: list,
    ):
        self.prompt = self.prompt.replace("<technology>", self.technology)
        self.prompt = self.prompt.replace("<general_concept>", self.general_concept)

        self.prompt = self.prompt.replace("<query>", query)
        self.prompt = self.prompt.replace(
            "<description>", json.dumps(description, ensure_ascii=False)
        )
        self.prompt = self.prompt.replace(
            "<papers>", self.research_papers_string(papers)
        )

        for n_topics in tqdm(
            self.split_number_with_interval(self.n_topics),
            desc="GENERATE MAIN CONCEPTS",
        ):
            prompt = self.prompt
            self.generate_general_topics_once(n_topics=n_topics)
            pprint(self.data.get_data())
            self.save_data()
            self.prompt = prompt

    def get_main_concepts(self, query: str):
        description = self.enhance_description(query=query)
        papers = self.get_research_papers(description=description)
        self.generate_main_concepts(
            query=query,
            description=description,
            papers=papers,
        )

    def generate(self, technology: str, general_concept: str, query: str, n_topics=10):
        self.technology = technology
        self.n_topics = n_topics
        self.general_concept = general_concept
        self.get_main_concepts(query=query)


def test_gen_general():
    client = GENERATE_GENERAL_TOPIC()
    query = "What are comprehensive concepts on the technologies"
    technology = "SDGs"
    client.generate(technology=technology, query=query, n_topics=20)


def test_query_gen():
    client = GENERATE_ENHANCE_QUERY()
    technology = "SAF"
    query = "SAF周りにどういった材料や、資源、活用法 があるのか網羅的に見たい"

    print("Result")
    client.generate(
        technology=technology,
        query=query,
        number_of_query=20,
    )
    pprint(client.data.get_data())


def test_gen_main():
    client = GENERATE_MAIN_TOPIC()
    technology = "SAF"
    general_concept = "SAF"
    query = "What are comprehensive concepts on the technologies, resources, and applications surrounding SAF, which plays an important role in efforts to reduce carbon emissions and move toward a more sustainable future."
    client.generate(
        technology=technology,
        general_concept=general_concept,
        query=query,
        n_topics=50,
    )


def main():
    # test_gen()
    test_query_gen()
    # test_gen_main()


if __name__ == "__main__":
    main()

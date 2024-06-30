import dotenv

from llm import LLM
from typing import Literal
from dbclient import DBClient, FILTERS, Papers
import os
import json
from pprint import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm

dotenv.load_dotenv()


class FACT_CHECK(DBClient, LLM):

    PROMPT_DIR = "./prompt"
    PROMPT_FILE = "enhance_vectordb_query.txt"

    BASE_DIR = "./data"
    SAVE_DIR = "./generated_enhance_query"

    RESULT_SAVE_DIR = "fact_check"

    def __init__(
        self,
        use_model: Literal[
            "gpt3",
            "gpt4",
            "claude1",
            "claude2",
            "claude2-1",
            "titan-express",
            "titan-lite",
            "jurassic-ultra",
        ] = "gpt4",
        base: Literal[
            "openai",
            "azure",
            "bedrock",
        ] = "azure",
        temperature=0,
        max_tokens=2000,
        top_p=1,
        n_result=20,
        save=True,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.save = save

        super().__init__(n_result=n_result)
        LLM.__init__(
            self,
            prompt_file=f"{self.PROMPT_DIR}/{self.PROMPT_FILE}",
            use_model=use_model,
            base=base,
        )

    def enhance_query(self, query: str):
        if len(query) < 2:
            return ""

        prompt = self.prompt
        self.prompt = self.prompt.replace("<query>", query)
        result = self.get_response(
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        result = json.loads(result)["result"]
        self.prompt = prompt
        return result

    def enhance_queries(self, queries: list[str]):
        return [self.enhance_query(query) for query in queries]

    def format_papers_list(self):
        paper_citation_string = (
            lambda title, abstract, publication_year, score, doi: f"{title} ({publication_year}), Score: {score}, Doi: ({doi})"
        )
        return "\n".join(
            [
                paper_citation_string(
                    paper.title,
                    paper.abstract,
                    paper.publication_year,
                    paper.score,
                    paper.doi,
                )
                for paper in self.papers_result.data
            ]
        )

    def get_similar(self, query: str):
        return self.get_papers(
            query=query,
            method="search",
            keyword_filter=self.filters,
            with_vector=self.with_vector,
        )

    def get_recommend(
        self,
        positive_query: list[str],
        negative_query: list[str],
        score: Literal["best_score", "average_vector"],
    ):
        return self.get_papers(
            negative_query=negative_query,
            positive_query=positive_query,
            method="explore",
            keyword_filter=self.filters,
            score=score,
            with_vector=self.with_vector,
        )

    def get_discover(
        self,
        query: str,
        pos_neg_pairs: list[(str, str)] = [],
    ):
        return self.get_papers(
            query=query,
            pos_neg_pairs=pos_neg_pairs,
            method="discover",
            keyword_filter=self.filters,
            with_vector=self.with_vector,
        )

    def save_papers(self, query: str):
        file_dir = f"{self.BASE_DIR}/{self.RESULT_SAVE_DIR}/{query}"
        os.makedirs(file_dir, exist_ok=True)
        self.papers_result.save_data(path=f"{file_dir}/papers.csv")

    def save_vectors(self, query: str):
        file_dir = f"{self.BASE_DIR}/{self.RESULT_SAVE_DIR}/{query}"
        os.makedirs(file_dir, exist_ok=True)
        self.papers_result.save_vectors(path=f"{file_dir}/vectors.npz")

    def check_query(
        self,
        query: str = "",
        method: Literal["search", "explore", "discover"] = "search",
        score: Literal["best_score", "average_vector"] = "best_score",
        positive_query: list[str] = [],
        negative_query: list[str] = [],
        pos_neg_pairs: list[list[str]] = [],
        should_keyword: list[dict] = [],
        must_keyword: list[dict] = [],
        must_not_keyword: list[dict] = [],
        with_enhance=False,
        with_vector=False,
    ):

        save_query = "_".join(query.split())
        self.with_vector = with_vector
        self.filters = FILTERS(
            should=should_keyword,
            must=must_keyword,
            must_not=must_not_keyword,
        )

        if method == "search":
            if with_enhance:
                enhanced_query = self.enhance_query(query)
                query = f"{query} {enhanced_query}"
                print(query)

            papers = self.get_similar(query=query)
        elif method == "explore":
            if score == "average_vector":
                assert len(positive_query) > 0, "At least one positive query"

            save_query = (
                f"pos={'+'.join(positive_query)}_neg={'+'.join(negative_query)}"
            )
            if with_enhance:
                enhanced_positive_query = self.enhance_queries(positive_query)
                enhanced_negative_query = self.enhance_queries(negative_query)
                positive_query = [
                    f"{query} {enhanced_query}"
                    for query, enhanced_query in zip(
                        positive_query, enhanced_positive_query
                    )
                ]
                negative_query = [
                    f"{query} {enhanced_query}"
                    for query, enhanced_query in zip(
                        negative_query, enhanced_negative_query
                    )
                ]

                pprint(positive_query)
                pprint(negative_query)

            papers = self.get_recommend(
                positive_query=positive_query,
                negative_query=negative_query,
                score=score,
            )
        elif method == "discover":
            if with_enhance:
                enhanced_query = self.enhance_query(query)
                query = f"{query} {enhanced_query}"

                enhanced_pos_neg_pairs = [
                    self.enhance_queries(pair) for pair in pos_neg_pairs
                ]

                pos_neg_pairs = [
                    [
                        f"{pos} {enhanced_pos}",
                        f"{neg} {enhanced_neg}",
                    ]
                    for (pos, neg), (enhanced_pos, enhanced_neg) in zip(
                        pos_neg_pairs, enhanced_pos_neg_pairs
                    )
                ]

            papers = self.get_discover(
                query=query,
                pos_neg_pairs=pos_neg_pairs,
            )

        self.papers_result: Papers = papers
        if self.save:
            self.save_papers(save_query)
            if with_vector:
                self.save_vectors(save_query)

        # Debug用
        # print(len(papers.get_data()))
        # tqdm.write(self.format_papers_list())
        # print()


# 類似度検索
def test_fact_check():
    fact_check_client = FACT_CHECK(
        n_result=20,
        save=False,
        max_tokens=4000,
    )
    query = "Application of LiNGAM"
    or_keywords = [
        # {"abstract": "LiNGAM"},
    ]  # [{"title": "LiNGAM"}]
    and_keywords = [
        # {"title": "LiNGAM"},
    ]  # {"title": "fuel"}
    not_keywords = []  # [{"title": "Fuel"}, {"title": "fuel"}]  # {"title": "fuel"}

    fact_check_client.check_query(
        query,
        method="search",
        should_keyword=or_keywords,
        must_keyword=and_keywords,
        must_not_keyword=not_keywords,
        with_enhance=True,
        with_vector=False,
    )
    # fact_check_client.papers_result.get_data()


# 　曖昧検索
def test_recommend_check():
    fact_check_client = FACT_CHECK(n_result=20, save=False)

    or_keywords = []  # [{"title": "Fuel"}]
    and_keywords = [
        # {"title": "Augmented"},
        # {"title": "Reality"},
    ]  # {"title": "fuel"}
    not_keywords = []  # [{"title": "Fuel"}, {"title": "fuel"}]  # {"title": "fuel"}

    positive_query = [
        "Using LLM for Ideation in Augmented Reality",
        # "Ideation in Virtual and Augmented Reality",
        "Augemented Reality as Ideation tools",
        "Using LLMs in Augmented Reality",
        # "Research on next-generation ideation tools using augmented reality (AR) and generative AI. ",
        # "This paper aims to explore a generative approach for knowledge-based design ideation by applying the latest pre-trained language models in artificial intelligence (AI). Specifically, a method of fine-tuning the generative pre-trained transformer using the USPTO patent database is proposed. The AI-generated ideas are not only in concise and understandable language but also able to synthesize the target design with external knowledge sources with controllable knowledge distance. The method is tested in a case study of rolling toy design and the results show good performance in generating ideas of varied novelty with near-field and far-field source knowledge.",
    ]
    negative_query = [
        "共創は、市場投入のスピードアップ、製品品質の向上、市場失敗リスクの低減といったビジネス上のメリットをもたらす。しかし、共創的なデザイン・セッションは、デザイナーと非デザイナーとの間のコミュニケーションの障壁によって、誤解が生じたり、効率的なアイデアの交換が阻害されたりするため、困難な場合があります。このような課題を克服し、より効果的な共創セッションをサポートする拡張現実ベースのデザイン表現の可能性を、プロのデザイナーとエンドユーザーを対象に実施した対照実験を通して探る。"
    ]

    fact_check_client.check_query(
        positive_query=positive_query,
        negative_query=negative_query,
        method="explore",
        score="best_score",
        should_keyword=or_keywords,
        must_keyword=and_keywords,
        must_not_keyword=not_keywords,
        with_enhance=True,
        with_vector=False,
    )


# グラフ検索
def test_discover_check():
    fact_check_client = FACT_CHECK(
        use_model="claude1",
        max_tokens=700,
        save=True,
    )

    or_keywords = []  # [{"title": "Fuel"}]
    and_keywords = [
        # {"title": "Preliminaries"},
    ]  # {"title": "fuel"}
    not_keywords = [
        # {"title": "volume"},
        # {"title": "Preliminaries"},
    ]  # [{"title": "Fuel"}, {"title": "fuel"}]  # {"title": "fuel"}

    query = "Using Large Language Models in Augmented Reality(AR) Ideation Tools"
    pos_neg_pair = [
        ["Ideation Tools", "co-designing"],
        ["GPT as Tools", "Image Generation as Tools"],
        # ["Augmented Reality as Ideation Tools", "Generative AI "],
    ]

    fact_check_client.check_query(
        query=query,
        pos_neg_pairs=pos_neg_pair,
        method="discover",
        should_keyword=or_keywords,
        must_keyword=and_keywords,
        must_not_keyword=not_keywords,
        with_enhance=False,
        with_vector=False,
    )


def main():
    test_fact_check()
    # test_recommend_check()
    # test_discover_check()
    pass


if __name__ == "__main__":
    main()

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    SearchParams,
    SearchRequest,
    RecommendStrategy,
    RecommendRequest,
    Filter,
    FieldCondition,
    MatchText,
    DiscoverRequest,
    ContextExamplePair,
)
from typing import Literal
import dotenv
import os
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np
from tqdm import tqdm
from time import sleep
import pandas as pd
from pprint import pprint

dotenv.load_dotenv()
from datetime import datetime


class Specter2Embedder:

    BASE_MODEL = "allenai/specter2_base"
    EMBED_ADAPTER = "allenai/specter2"
    QUERY_ADAPTER = "allenai/specter2_adhoc_query"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # defaults to loading the query adapter
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.decoder = AutoAdapterModel.from_pretrained(self.BASE_MODEL)

        self.load()

    def load(self):
        self.decoder.load_adapter(
            self.EMBED_ADAPTER,
            source="hf",
            load_as="specter2_query",
            set_active=True,
        )
        self.decoder.to(self.device)

    def encode(self, texts: list[str]) -> list[str]:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.decoder(**inputs)
            output = output.last_hidden_state[:, 0, :]
            return output.cpu().numpy().tolist()

    def encode_text(self, text: str) -> list[list]:
        return self.encode(texts=[text])[0]

    def encode_texts(self, texts: list[str]) -> list[str]:
        if texts != []:
            return self.encode(texts=texts)


class Paper:
    def __init__(
        self,
        id: int,
        title: str,
        abstract: str,
        publication_date: str,
        concepts: dict,
        score: float,
        doi: str,
        cited_by_count: int,
        authors: list,
        vector: np.ndarray[1, 768] = None,
    ) -> None:
        self.id = id
        self.title = title
        self.abstract = abstract
        self.concepts = concepts  # TODO: Add Concept Class
        self.score = round(score, 4)
        self.doi = doi
        self.cited_by_count = cited_by_count
        self.publication_date = datetime.strptime(publication_date, "%Y-%m-%d")
        self.publication_year = self.publication_date.year
        # TODO: Add counts by year
        # TODO: Add authors
        self.authors = authors
        self.vector = vector

    def get_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "publication_date": str(self.publication_date.date()),
            "publication_year": self.publication_date.year,
            "concepts": self.concepts,
            "score": self.score,
            "doi": self.doi,
            "cited_by_count": self.cited_by_count,
            "authors": self.authors,
        }


class Papers:
    def __init__(self, paper_list: list) -> None:
        self.paper_list = paper_list
        self.data = list[Paper](self.read_paper_list())
        pass

    def read_paper_list(self):
        return [
            Paper(
                id=paper["id"],
                title=paper["title"],
                abstract=paper["abstract"],
                publication_date=paper["publication_date"],
                concepts=paper["concepts"],
                score=paper["score"],
                doi=paper["doi"],
                cited_by_count=paper["cited_by_count"],
                vector=paper["vector"],
                authors=paper["authorships"],
            )
            for paper in self.paper_list
        ]

    def get_data(self):
        return [paper.get_dict() for paper in self.data]

    def save_data(self, path: str):
        columns = [
            "id",
            "title",
            "abstract",
            "publication_date",
            "publication_year",
            "concepts",
            "score",
            "doi",
            "cited_by_count",
        ]
        data = {
            col: [" ".join(str(item[col]).split()) for item in self.get_data()]
            for col in columns
        }
        df = pd.DataFrame(data, columns=columns)

        df.to_csv(path, index=False)

    def get_vector(self):
        sorted_papers = sorted(self.data, key=lambda paper: paper.score, reverse=True)

        return [paper.vector for paper in sorted_papers]

    def save_vectors(self, path: str):
        vector_data = {data.id: data.vector for data in self.data}
        np.savez(path, **vector_data)


class FILTER_VALUE:
    def __init__(self, key="", value=""):
        self.key = key
        self.value = value


class FILTER:
    def __init__(self, filter_list: list[dict]):
        self.filter_list = filter_list
        self.data = list[FILTER_VALUE](self.read_filter_list())

    def read_filter_list(self):
        data = []
        for filter in self.filter_list:
            key, value = next(iter(filter.items()))
            data.append(FILTER_VALUE(key=key, value=value))
        return list[FILTER_VALUE](data)


class FILTERS:

    def __init__(self, should=[], must=[], must_not=[]):
        self.should = FILTER(should)
        self.must = FILTER(must)
        self.must_not = FILTER(must_not)


class DBClient(Specter2Embedder):
    SPLIT_NUM = 1000

    def __init__(self, n_result=100) -> None:
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.retrieve_load_list = [
            "title",
            "doi",
            "cited_by_count",
            "counts_by_year",
            "publication_date",
            "abstract",
            "concepts",
            "authorships",
        ]
        self.n_result = n_result
        super().__init__()

    def split_number_with_interval(self, total: int):
        return [self.SPLIT_NUM] * (total // self.SPLIT_NUM) + [total % self.SPLIT_NUM]

    def create_filter(self, filters: FILTER):
        return [
            FieldCondition(
                key=item.key,
                match=MatchText(
                    text=item.value,
                ),
            )
            for item in filters.data
        ]

    def create_keyword_filter(self, filters: FILTERS):
        return Filter(
            should=self.create_filter(filters.should),
            must=self.create_filter(filters.must),
            must_not=self.create_filter(filters.must_not),
        )

    def create_search_request(
        self,
        filters: dict,
        query_vector: np.ndarray[1, 768],
        with_vector: bool,
    ):
        return list[SearchRequest](
            [
                SearchRequest(
                    params=SearchParams(hnsw_ef=128, exact=False),
                    vector=query_vector,
                    filter=self.create_keyword_filter(filters),
                    limit=value,
                    offset=(self.SPLIT_NUM * index) + 1,
                    with_payload=True,
                    with_vector=with_vector,
                )
                for index, value in enumerate(
                    self.split_number_with_interval(self.n_result)
                )
                if value
            ]
        )

    def search_document(
        self,
        target_collection: str,
        filters: dict,
        query_vector: np.ndarray[1, 768],
        with_vector: bool,
        consistency: Literal["all", "majority", "quorum"] = "all",
    ):
        requests = self.create_search_request(
            filters=filters,
            query_vector=query_vector,
            with_vector=with_vector,
        )

        result = self.qdrant_client.search_batch(
            collection_name=target_collection,
            requests=requests,
            consistency="all",
        )
        return [paper for papers in result for paper in papers]

    def create_explore_request(
        self,
        filters: FILTERS,
        with_vector: bool,
        positive_query_vectors: list[np.ndarray[1, 768]],
        negative_query_vectors: list[np.ndarray[1, 768]],
    ):
        return list[RecommendRequest](
            [
                RecommendRequest(
                    params=SearchParams(hnsw_ef=128, exact=False),
                    filter=self.create_keyword_filter(filters),
                    positive=positive_query_vectors,
                    negative=negative_query_vectors,
                    limit=value,
                    offset=(self.SPLIT_NUM * index) + 1,
                    with_payload=True,
                    with_vector=with_vector,
                )
                for index, value in enumerate(
                    self.split_number_with_interval(self.n_result)
                )
                if value
            ]
        )

    def explore_document(
        self,
        target_collection: str,
        filters: FILTERS,
        positive_query_vectors: list[np.ndarray[1, 768]],
        negative_query_vectors: list[np.ndarray[1, 768]],
        score: Literal["best_score", "average_vector"],
        with_vector: bool,
        consistency: Literal["all", "majority", "quorum"] = "all",
    ):
        if score == "best_score":
            score = RecommendStrategy.BEST_SCORE
        elif score == "average_vector":
            score = RecommendStrategy.AVERAGE_VECTOR

        requests = self.create_explore_request(
            filters=filters,
            with_vector=with_vector,
            positive_query_vectors=positive_query_vectors,
            negative_query_vectors=negative_query_vectors,
        )

        result = self.qdrant_client.recommend_batch(
            collection_name=target_collection,
            requests=requests,
            consistency=consistency,
        )
        return [paper for papers in result for paper in papers]

    def create_discover_request(
        self,
        filters: FILTERS,
        target: np.ndarray[1, 768],
        with_vector: bool,
        pairs: list[(np.ndarray[1, 768], np.ndarray[1, 768])] = [],
    ):
        context_pairs = [
            ContextExamplePair(
                positive=negative,
                negative=positive,
            )
            for positive, negative in pairs
        ]
        return list[DiscoverRequest](
            [
                DiscoverRequest(
                    params=SearchParams(hnsw_ef=128, exact=False),
                    target=target,
                    context=context_pairs,
                    filter=self.create_keyword_filter(filters),
                    limit=value,
                    offset=(self.SPLIT_NUM * index) + 1,
                    with_payload=True,
                    with_vector=with_vector,
                )
                for index, value in enumerate(
                    self.split_number_with_interval(self.n_result)
                )
                if value
            ]
        )

    def discover_document(
        self,
        target_collection: str,
        filters: FILTERS,
        target: np.ndarray[1, 768],
        with_vector: bool,
        pairs: list[(np.ndarray[1, 768], np.ndarray[1, 768])] = [],
        consistency: Literal["all", "majority", "quorum"] = "all",
    ):
        requests = self.create_discover_request(
            filters=filters,
            target=target,
            with_vector=with_vector,
            pairs=pairs,
        )
        result = self.qdrant_client.discover_batch(
            collection_name=target_collection,
            requests=requests,
            consistency=consistency,
        )
        return [paper for papers in result for paper in papers]

    def get_papers(
        self,
        query: str = "",
        positive_query: list[str] = [],
        negative_query: list[str] = [],
        keyword_filter: FILTERS = FILTERS(),
        pos_neg_pairs: list[(str, str)] = [],
        score: Literal["best_score", "average_vector"] = "best_score",
        method: Literal["search", "explore", "discover"] = "search",
        with_vector: bool = False,
    ):
        self.query = query

        if method == "search":
            query_encoded = self.encode_text(query)
            result = self.search_document(
                target_collection="work",
                filters=keyword_filter,
                query_vector=query_encoded,
                with_vector=with_vector,
            )
        elif method == "explore":
            positive_query_encoded = self.encode_texts(positive_query)
            negative_query_encoded = self.encode_texts(negative_query)
            result = self.explore_document(
                target_collection="work",
                score=score,
                filters=keyword_filter,
                positive_query_vectors=positive_query_encoded,
                negative_query_vectors=negative_query_encoded,
                with_vector=with_vector,
            )
        elif method == "discover":
            query_encoded = self.encode_text(query)
            pos_neg_pairs_encoded = [self.encode_texts(pair) for pair in pos_neg_pairs]
            result = self.discover_document(
                target_collection="work",
                target=query_encoded,
                filters=keyword_filter,
                pairs=pos_neg_pairs_encoded,
                with_vector=with_vector,
            )

        papers = [
            {
                "id": str(hit.id),
                "score": hit.score,
                "vector": hit.vector if hit.vector != None else None,
                **{load: hit.payload[load] for load in self.retrieve_load_list},
            }
            for hit in result
        ]
        return Papers(papers)


def test():
    client = DBClient(20)
    query = "Optogenetics, leveraging the features of LEDs such as energy efficiency and precision in controlling light intensity, has revolutionized drug development and research within the pharmaceutical and biotechnology market. The problem of inadequate research models, which has hindered the understanding and testing of new drugs, is being addressed by optogenetics. This technology functions by using light to control cells in living tissue, typically neurons, which allows researchers to better simulate and study complex biological processes in a controlled environment."
    client.get_papers(query, method="search")


def main():
    test()


if __name__ == "__main__":
    main()

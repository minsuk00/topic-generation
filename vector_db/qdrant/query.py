from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
import dotenv
import os
import torch
import time
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import openai


class Specter2Embedder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.BASE_MODEL = "allenai/specter2_base"
        self.EMBED_ADAPTER = "allenai/specter2"
        self.QUERY_ADAPTER = "allenai/specter2_adhoc_query"
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL)
        # defaults to loading the query adapter
        self.model.load_adapter(
            self.EMBED_ADAPTER, source="hf", load_as="specter2_query", set_active=True
        )
        self.model.to(self.device)

    def encode(self, texts: list[str]) -> list[list]:
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
            output = self.model(**inputs)
            output = output.last_hidden_state[:, 0, :]
            return output.cpu().numpy().tolist()


def init():
    dotenv.load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )
    se = Specter2Embedder()
    return client, se


def enhance_query(query, print_enhanced=False):
    response = (
        openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who is a professional academic writer.",
                },
                {
                    "role": "user",
                    "content": f"INSTRUCTIONS: write a concise informational sentence about the following query. Use as many relevant keywords as possible to explain the semantics of this topic. \n\nQuery: {query}",
                },
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        .choices[0]
        .message.content
    )

    if print_enhanced:
        print(f"enhanced response: {response}")
    return response


def main():
    client, se = init()
    enhance = True
    while True:
        query = input("Enter query: ")
        if enhance:
            query = enhance_query(query, print_enhanced=True)
        vector = se.encode([query])[0]
        before_time = time.time()
        res = client.search(
            collection_name="work",
            search_params=SearchParams(hnsw_ef=128, exact=False),
            query_vector=vector,
            limit=10,
        )
        # print time in ms
        print(f"Qdrant Search Time: {(time.time() - before_time) * 1000}ms")

        qdrant_res = [
            {
                "shorten_id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in res
        ]

        for q_res in qdrant_res:
            print("   score: ", q_res["score"])
            print("   title: ", q_res["payload"]["title"])
            print("   doi: ", q_res["payload"]["doi"])
            print("   abstract: ", q_res["payload"]["abstract"])
            print("\n\n")


if __name__ == "__main__":
    main()

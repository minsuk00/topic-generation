from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    Batch,
    PointStruct,
)
import dotenv
import os
import glob
import json

# create type


def get_all_json_files(directory: str = "vectors/hybrid"):
    return sorted(glob.glob(f"{directory}/*.json", recursive=False))


# return the json file in-memory
def load_json_file(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def batch_insert(
    client: QdrantClient,
    collection_name: str,
    batch: list[dict],
    file_name: str,
    batch_index: int,
):
    # retry 3 times
    retry_count = 0
    while retry_count < 3:
        try:
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=int(vec["shorten_id"]),
                        vector={
                            "dense": vec["dense_vector"],
                            "sparse": models.SparseVector(
                                indices=vec["sparse_vector"]["indices"],
                                values=vec["sparse_vector"]["values"],
                            ),
                        },
                        payload=vec["payloads"],
                    )
                    for vec in batch
                ],
                wait=True,
            )
            return
        except Exception as e:
            print(e)
            retry_count += 1
            print(
                f"Retrying... file_name: {file_name}, batch_index: {batch_index}, retry_count: {retry_count}"
            )


def get_next_batch(all_data: list, batch_size: int = 1000) -> list:
    buffer = []
    for data in all_data:
        buffer.append(data)
        # バッファのサイズがbatch_sizeに達したら、バッファの内容をyield
        if len(buffer) == batch_size:
            yield buffer
            buffer = []
    # 最後に残ったデータを返す
    if buffer:
        yield buffer


def insert_from_json_file(client: QdrantClient, file_name: str, collection_name: str):
    data = load_json_file(file_name)
    batch_size = 1000
    for i, batch in enumerate(get_next_batch(data, batch_size)):
        print(f"inserting {i}th batch")
        batch_insert(client, collection_name, batch, file_name, i)


def main(collection_name: str = "new_small_work"):
    dotenv.load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=20
    )

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
    except Exception as e:
        print(e)
        print("collection already exists")

    for vec_json_path in get_all_json_files():
        print(f"inserting {vec_json_path}...")
        insert_from_json_file(client, vec_json_path, collection_name)


if __name__ == "__main__":
    main()

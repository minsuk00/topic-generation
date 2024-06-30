import glob
import numpy as np
import json
import logging
import torch
import multiprocessing
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from accelerate import Accelerator


class Specter2Embedder:
    def __init__(self, accelerator: Accelerator):
        self.BASE_MODEL = "allenai/specter2_base"
        self.EMBED_ADAPTER = "allenai/specter2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL)
        self.model.load_adapter(
            self.EMBED_ADAPTER, source="hf", load_as="specter2", set_active=True
        )
        self.accelerator = accelerator
        self.model.to(self.accelerator.device)

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        with self.accelerator.split_between_processes(texts) as prompts:
            with torch.no_grad():
                # Process the prompts with the tokenizer and model
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512,
                ).to(self.accelerator.device)
                outputs = self.model(**inputs)
                outputs = outputs.last_hidden_state[:, 0, :]

        # Use accelerator.gather to collect outputs from all processes
        gathered_outputs = self.accelerator.gather(outputs)

        # Ensure the output handling (e.g., moving to CPU, converting to NumPy) is done on the main process
        if self.accelerator.is_main_process:
            gathered_outputs = gathered_outputs.cpu().numpy()
            return [
                gathered_outputs[i].reshape(1, -1)
                for i in range(gathered_outputs.shape[0])
            ]
        else:
            # For non-main processes, you could return an empty list or handle as needed
            return []


def inverted_index_to_paragraph(inverted_index: dict):
    # Create a paragraph
    paragraph = []
    # Iterate through the words in the inverted index
    for word, _ in sorted(inverted_index.items(), key=lambda x: x[1][0]):
        paragraph.append(word)
    # Join the words to form a paragraph
    paragraph_text = " ".join(paragraph)
    return paragraph_text


def get_all_json_files(directory: str):
    return glob.glob(f"{directory}/*.json", recursive=False)


def save_as_json(works_id_vector_json, offset: int):
    with open(f"vectors/batch_{offset}.json", "w", encoding="utf-8") as fp:
        json.dump(works_id_vector_json, fp)


def remove_prefix(s: str, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :] if s.startswith(prefix) else s
    return s


## from_s3/**.jsonからchunk_sizeだけ読み出して配列にして返す
def get_next_works_chunk(start_line, chunk_size, file_list: list[str]):
    print(file_list)
    print(f"skipping first {start_line} lines")
    buffer = []
    for file_path in file_list:
        with open(
            file_path, "rt", encoding="utf-8"
        ) as file:  # 'rt'モードでファイルを開く
            for line in file:
                if start_line > 0:
                    start_line -= 1
                    continue
                # JSONをパースしてバッファに追加
                try:
                    json_obj = json.loads(line)
                    abstract = inverted_index_to_paragraph(
                        json_obj["abstract_inverted_index"]
                    )
                    json_obj = {
                        "shorten_id": remove_prefix(
                            json_obj["id"], "https://openalex.org/W"
                        ),
                        "payloads": {
                            "title": json_obj["title"],
                            "abstract": abstract,
                            "doi": json_obj["doi"],
                            "authorships": [
                                {
                                    "author": a["author"]["display_name"],
                                    "affiliation": a["raw_affiliation_string"],
                                    "position": a["author_position"],
                                }
                                for a in json_obj["authorships"]
                            ],
                            "concepts": [
                                {
                                    "name": c["display_name"],
                                    "level": c["level"],
                                }
                                for c in json_obj["concepts"]
                            ],
                            "cited_by_count": json_obj["cited_by_count"],
                            "publication_date": json_obj["publication_date"],
                            "counts_by_year": json_obj["counts_by_year"],
                        },
                    }
                    buffer.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"JSONデコードエラー: {e}, ファイル: {file_path}, 行: {line}")
                # バッファのサイズがchunk_sizeに達したら、バッファの内容をyield
                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []
    # 最後に残ったデータを返す
    if buffer:
        yield buffer


def main():
    logging.basicConfig(filename="transform.log", encoding="utf-8", level=logging.INFO)
    TOTAL_WORKS_COUNT = 31320433  ## SQLで数えた
    CHUNK_SIZE = 400 * 4
    START_PAGE = 0
    START_LINE = START_PAGE * CHUNK_SIZE
    DROPPED_COUNT = 0
    actual_total_count = 0

    emb = Specter2Embedder(Accelerator())

    logging.info("TOTAL WORKS COUNT : " + str(TOTAL_WORKS_COUNT))

    pool = multiprocessing.Pool(processes=2)
    file_list = get_all_json_files("from_s3")

    works_id_vector_json = []
    batch_counter = 0
    for i, works in enumerate(
        get_next_works_chunk(START_LINE, CHUNK_SIZE, file_list), START_PAGE
    ):
        logging.info(
            f"==================== TotalProcessedCount... {CHUNK_SIZE * i} / {TOTAL_WORKS_COUNT}  -- ({100* CHUNK_SIZE * i/TOTAL_WORKS_COUNT}% done) ===================="
        )

        # word count <= 20 のものは除外
        original = len(works)
        actual_total_count += original
        works = [w for w in works if len(w["payloads"]["abstract"].split()) > 20]
        DROPPED_COUNT += original - len(works)
        logging.info(f"actual_total_count: {actual_total_count}")
        logging.info(f"DROPPED_COUNT: {DROPPED_COUNT}")
        logging.info(f"drop percentage: {100*DROPPED_COUNT/actual_total_count} %")

        # title + abstract を結合してベクトル化
        vectors = emb.encode(
            [w["payloads"]["title"] + " " + w["payloads"]["abstract"] for w in works]
        )

        # サブプロセスのときは何もしない
        if len(vectors) == 0:
            continue

        # WORKS IDとベクトルを紐付けてjsonにする
        for work, vector in zip(works, vectors):
            works_id_vector_json.append(
                {
                    "shorten_id": work["shorten_id"],
                    "vector": vector.tolist(),
                    "payloads": work["payloads"],
                }
            )
        if len(works_id_vector_json) >= 100000:
            # 重たいので別プロセスで実行する
            pool.apply_async(save_as_json, (works_id_vector_json, batch_counter))
            works_id_vector_json = []
            batch_counter += 1
        logging.info(f"len of works_id_vector_json: {len(works_id_vector_json)}")
        logging.info("==================== DONE ====================\n\n\n")
    if works_id_vector_json:
        pool.apply_async(save_as_json, (works_id_vector_json, batch_counter))


if __name__ == "__main__":
    main()

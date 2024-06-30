import glob
import numpy as np
import json
import logging
import torch
import multiprocessing
from typing import Literal
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from accelerate import Accelerator
from dotenv import load_dotenv

load_dotenv()


class SPLADEEmbedder:
    def __init__(self, accelerator: Accelerator):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "naver/splade-v3", token="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"
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            "naver/splade-v3", token="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"
        )
        self.accelerator = accelerator
        self.model.to(self.accelerator.device)

    def extract_and_map_sparse_vector(
        self, vector
    ):  # No need to use for embedding into Qdrant
        """
        Extracts non-zero elements from a given vector and maps these elements to their human-readable tokens using a tokenizer. The function creates and returns a sorted dictionary where keys are the tokens corresponding to non-zero elements in the vector, and values are the weights of these elements, sorted in descending order of weights.

        This function is useful in NLP tasks where you need to understand the significance of different tokens based on a model's output vector. It first identifies non-zero values in the vector, maps them to tokens, and sorts them by weight for better interpretability.

        Args:
        vector (torch.Tensor): A PyTorch tensor from which to extract non-zero elements.
        tokenizer: The tokenizer used for tokenization in the model, providing the mapping from tokens to indices.

        Returns:
        dict: A sorted dictionary mapping human-readable tokens to their corresponding non-zero weights.
        """

        # Extract indices and values of non-zero elements in the vector
        cols = vector.nonzero().squeeze().cpu().tolist()
        weights = vector[cols].cpu().tolist()

        # Map indices to tokens and create a dictionary
        idx2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}
        token_weight_dict = {
            idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }

        # Sort the dictionary by weights in descending order
        sorted_token_weight_dict = {
            k: v
            for k, v in sorted(
                token_weight_dict.items(), key=lambda item: item[1], reverse=True
            )
        }

        return sorted_token_weight_dict

    def encode(self, texts: list[str]) -> list[tuple]:
        with self.accelerator.split_between_processes(texts) as prompts:
            with torch.inference_mode():
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512,
                ).to(self.accelerator.device)
                outputs = self.model(**inputs)
                logits, attention_mask = outputs.logits, inputs.attention_mask
                relu_log = torch.log(1 + torch.relu(logits))
                weighted_log = relu_log * attention_mask.unsqueeze(-1)
                max_val, _ = torch.max(weighted_log, dim=1)
                outputs = max_val.squeeze()

        # Use accelerator.gather to collect outputs from all processes
        gathered_outputs = self.accelerator.gather(outputs)

        # Ensure the output handling (e.g., moving to CPU, converting to NumPy) is done on the main process
        if self.accelerator.is_main_process:
            gathered_outputs = gathered_outputs.cpu().numpy()
            ind_val_results = []
            for i in range(gathered_outputs.shape[0]):
                nonzero_indices = gathered_outputs[i].nonzero()[0].tolist()
                nonzero_values = gathered_outputs[i][nonzero_indices].tolist()
                ind_val_results.append((nonzero_indices, nonzero_values))
            return ind_val_results
        else:
            # For non-main processes, you could return an empty list or handle as needed
            return []


class Specter2EmbedderAugRefresh:
    """
    Defaults to document embedding
    """

    def __init__(self, accelerator: Accelerator):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/specter2_aug2023refresh_base"
        )
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_aug2023refresh_base"
        )
        self.model.load_adapter(
            "allenai/specter2_aug2023refresh",
            source="hf",
            load_as="specter2_proximity",
            set_active=True,
        )
        self.accelerator = accelerator
        self.model.to(self.accelerator.device)

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        with self.accelerator.split_between_processes(texts) as prompts:
            with torch.inference_mode():
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
    with open(f"vectors/hybrid/batch_{offset}.json", "w", encoding="utf-8") as fp:
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
    logging.basicConfig(filename="v3_hybrid.log", encoding="utf-8", level=logging.INFO)
    TOTAL_WORKS_COUNT = 31320433  ## SQLで数えた
    CHUNK_SIZE = 50
    START_PAGE = 0
    START_LINE = START_PAGE * CHUNK_SIZE
    DROPPED_COUNT = 0
    actual_total_count = 0

    emb_dense = Specter2EmbedderAugRefresh(Accelerator())
    emb_sparse = SPLADEEmbedder(Accelerator())
    SEP_TOKEN = emb_dense.tokenizer.sep_token

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
        texts = [
            w["payloads"]["title"] + SEP_TOKEN + w["payloads"]["abstract"]
            for w in works
        ]
        assert (
            emb_dense.accelerator.is_main_process
            == emb_sparse.accelerator.is_main_process
        )
        vectors_dense = emb_dense.encode(texts)
        vectors_sparse = emb_sparse.encode(texts)
        # サブプロセスのときは何もしない
        if not emb_dense.accelerator.is_main_process:
            continue
        assert len(vectors_dense) == len(vectors_sparse)

        # WORKS IDとベクトルを紐付けてjsonにする
        for work, dense_vec, sparse_vec in zip(works, vectors_dense, vectors_sparse):
            works_id_vector_json.append(
                {
                    "shorten_id": work["shorten_id"],
                    "dense_vector": dense_vec[0].tolist(),
                    "sparse_vector": {
                        "indices": sparse_vec[0],
                        "values": sparse_vec[1],
                    },
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
    if works_id_vector_json and emb_dense.accelerator.is_main_process:
        save_as_json(works_id_vector_json, batch_counter)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

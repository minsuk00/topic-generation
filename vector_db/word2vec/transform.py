import glob
import gzip
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging
import torch
import multiprocessing


class SpecterEmbedder:
    MODEL_NAME = "sentence-transformers/allenai-specter"
    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)
    def encode(self, texts: list[str]) -> list[np.ndarray[1, 768]]:
        return self.model.encode(sentences=texts,show_progress_bar=True,device=  "cuda" if torch.cuda.is_available() else "cpu")

def inverted_index_to_paragraph(inverted_index: dict):
    # Create a paragraph
    paragraph = []
    # Iterate through the words in the inverted index
    for word, _ in sorted(inverted_index.items(), key=lambda x: x[1][0]):
        paragraph.append(word)
    # Join the words to form a paragraph
    paragraph_text = ' '.join(paragraph)
    return paragraph_text


def get_all_json_files(directory:str):
    return glob.glob(f"{directory}/*.json", recursive=False)

def save_as_gzip(works_id_vector_json, offset:int):
    with gzip.open(f'vectors/batch_{offset}.json.gz', 'wt', encoding='utf-8') as fp:
        json.dump(works_id_vector_json, fp)


## from_s3/**.jsonからchunk_sizeだけ読み出して配列にして返す
def get_next_works_chunk(start_line, chunk_size,file_list:list[str]):
    print(file_list)
    print(f"skipping first {start_line} lines")
    buffer = []
    for file_path in file_list:
        with open(file_path, 'rt', encoding='utf-8') as file:  # 'rt'モードでファイルを開く
            for line in file:
                if start_line>0:
                    start_line -= 1
                    continue
                # JSONをパースしてバッファに追加
                try:
                    json_obj = json.loads(line)
                    json_obj = {
                        "id": json_obj["id"],
                        "title": json_obj["title"],
                        "abstract": inverted_index_to_paragraph(json_obj["abstract_inverted_index"])
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
    logging.basicConfig(filename='transform.log', encoding='utf-8', level=logging.INFO)
    TOTAL_WORKS_COUNT = 35896867 ## SQLで数えた
    CHUNK_SIZE = 100000 ## 10万ずつ変換する
    START_PAGE = 15
    START_LINE = START_PAGE * CHUNK_SIZE
    emb = SpecterEmbedder()
    logging.info("TOTAL WORKS COUNT : " + str(TOTAL_WORKS_COUNT))
    total_done_works = START_LINE
    pool = multiprocessing.Pool(processes=2)
    file_list = get_all_json_files("from_s3")
    for i, works in enumerate(get_next_works_chunk(START_LINE,CHUNK_SIZE, file_list),START_PAGE):
        logging.info(f"==================== TotalProcessedCount... {total_done_works} / {TOTAL_WORKS_COUNT}  -- ({100* total_done_works/TOTAL_WORKS_COUNT}% done) ====================")
        
        # title + abstract を結合してベクトル化
        vectors =  emb.encode([w["title"] + " " + w["abstract"] for w in works])
        
        # 作品IDとベクトルを紐付けてjsonにする
        works_id_vector_json = []
        for work, vector in zip(works, vectors):
            works_id_vector_json.append({"id":work["id"], "vector":vector.tolist()})
        
        # save as gzip
        # 重たいので別プロセスで実行する
        pool.apply_async(save_as_gzip, (works_id_vector_json, i))
        total_done_works += len(works)
        logging.info("==================== DONE ====================\n\n\n")

if __name__ == "__main__":
    main()

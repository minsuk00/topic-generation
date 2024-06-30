import json
from pprint import pprint
from get_papers import DBClient
from tqdm import tqdm
import pandas as pd
from ast import literal_eval


def topics():
    data = json.load(open("summaries.json", "r+"))
    children_parent = {item["child"]: item["parent"] for item in data}
    parents = list(set([parent for item in data for parent in item["parent"]]))
    children = list(set([item["child"] for item in data]))

    return children_parent, parents, children


def topics_summary():
    data = json.load(open("summaries.json", "r+"))
    return {item["child"]: item["summary"] for item in data}


def fetch_papers(data):
    client = DBClient(n_result=30)

    papers = {}
    for topic, summary in tqdm(list(data.items()), desc="Extract Child Papers"):
        fetched_papers, _, __ = client.get_paper(summary)
        papers[topic] = fetched_papers

    return papers


def create_node_data(topics: list, color: str, node_size: int, fontsize: int) -> dict:
    return [
        {
            "color": color,
            "font": {"color": "black", "face": "Times New Roman", "size": fontsize},
            "id": topic,
            "label": topic,
            "shape": "dot",
            "size": node_size,
            "position": {"x": True, "y": True},
        }
        for topic in topics
    ]


def create_node_parent_data(topics: list, node_size: int, fontsize: int) -> dict:
    colors = {
        "Algorithm": "#FF4D4D",  # Red (Science)
        "Education": "#FF9999",  # Light red
        "Criminology": "#FFCCCC",  # Lighter red
        "Emergency Management": "#FF9999",  # Light red
        "Computer Science": "#FF4D4D",  # Red (Science)
        "Art History": "#99CCFF",  # Light blue
        "Occupational Health": "#66FF66",  # Green
        "Economics": "#66CC99",  # Greenish
        "Textile Science": "#FF6666",  # Reddish
        "Ecology": "#33CC33",  # Green (Midway)
        "Health Informatics": "#33CC33",  # Green (Midway)
        "Geology": "#FF3333",  # Red (Science)
        "Public Health": "#66FF66",  # Green
        "Environmental Science": "#33CC33",  # Green (Midway)
        "Nuclear Physics": "#FF0000",  # Pure Red (Science)
        "Anthropology": "#99CCFF",  # Light blue
        "Materials Science": "#FF1A1A",  # Red (Science)
        "Zoology": "#33CC33",  # Green (Midway)
        "Agricultural Science": "#33CC33",  # Green (Midway)
        "Quantum Mechanics": "#FF0000",  # Pure Red (Science)
        "Evolutionary Biology": "#33CC33",  # Green (Midway)
        "Genetics": "#33CC33",  # Green (Midway)
        "Neuroscience": "#FF3333",  # Red (Science)
        "Statistical Physics": "#FF1A1A",  # Red (Science)
        "Finance": "#66CC99",  # Greenish
        "Business": "#66CC99",  # Greenish
        "Electrical Engineering": "#FF0000",  # Pure Red (Science)
        "Biochemistry": "#FF3333",  # Red (Science)
        "Medicine": "#66FF66",  # Green
        "Law": "#99CCFF",  # Light blue
        "Criminal Justice": "#99CCFF",  # Light blue
        "Physics": "#FF0000",  # Pure Red (Science)
        "Engineering": "#FF1A1A",  # Red (Science)
        "Gerontology": "#66FF66",  # Green
        "Surgery": "#66FF66",  # Green
        "Bioinformatics": "#33CC33",  # Green (Midway)
        "Chemistry": "#FF3333",  # Red (Science)
        "Cybersecurity": "#FF4D4D",  # Red (Science)
        "Microbiology": "#FF3333",  # Red (Science)
        "Telecommunications": "#33CC33",  # Green (Midway)
        "Electronics": "#FF1A1A",  # Red (Science)
        "Logistics": "#66CC99",  # Greenish
        "Urban Planning": "#99CCFF",  # Light blue
        "Pharmacology": "#66FF66",  # Green
        "Ethics": "#99CCFF",  # Light blue
        "Molecular Biology": "#33CC33",  # Green (Midway)
        "Nanotechnology": "#FF1A1A",  # Red (Science)
        "Astrophysics": "#FF0000",  # Pure Red (Science)
        "Photonics": "#FF1A1A",  # Red (Science)
        "Robotics": "#FF4D4D",  # Red (Science)
        "Psychology": "#66CC99",  # Greenish
        "Educational Research": "#FF9999",  # Light red
        "Data Science": "#FF4D4D",  # Red (Science)
        "Information Technology": "#FF4D4D",  # Red (Science)
        "Biomedical Engineering": "#FF0000",  # Pure Red (Science)
        "Cryptography": "#FF4D4D",  # Red (Science)
        "Mechanical Engineering": "#FF1A1A",  # Red (Science)
        "Machine Learning": "#FF4D4D",  # Red (Science)
        "Chemical Engineering": "#FF3333",  # Red (Science)
        "Oceanography": "#33CC33",  # Green (Midway)
        "Artificial Intelligence": "#FF4D4D",  # Red (Science)
        "Computer Vision": "#FF4D4D",  # Red (Science)
        "Linguistics": "#99CCFF",  # Light blue
        "Archaeology": "#99CCFF",  # Light blue
        "History": "#99CCFF",  # Light blue
        "Sustainable Development": "#66CC99",  # Greenish
    }
    return [
        {
            "color": colors[topic],
            "font": {"color": "black", "face": "Times New Roman", "size": fontsize},
            "id": topic,
            "label": topic,
            "shape": "dot",
            "size": node_size,
            "position": {"x": True, "y": True},
        }
        for topic in topics
    ]


def create_edges_data(topics_data: dict) -> dict:
    return [
        {
            "arrows": "from",
            "color": "#21918c",
            "font": {"color": "black", "face": "Times New Roman", "size": 10},
            "from": child,
            "to": parent,
            "label": "",
        }
        for child, parents in topics_data.items()
        for parent in parents
    ]


def create_child_relation_edges() -> dict:
    data = json.load(open("child_relation.json", "r"))
    return [
        {
            "arrows": "from",
            "color": "#21918c",
            "font": {"color": "black", "face": "Times New Roman", "size": 10},
            "to": item["from"],
            "from": item["to"],
            "label": item["relation"],
        }
        for item in data
    ]


def create_origin(parents):
    ORIGIN_NAME = "ORIGIN"
    origin_node = {
        "color": "white",
        "font": {"color": "black", "face": "Times New Roman", "size": 30},
        "id": ORIGIN_NAME,
        "label": ORIGIN_NAME,
        "shape": "dot",
        "size": 0,
    }

    origin_edges = [
        {
            "arrows": "from",
            "color": "#21918c",
            "font": {"color": "black", "face": "Times New Roman", "size": 0},
            "to": parent,
            "from": parent,
            "to": ORIGIN_NAME,
            "label": "",
        }
        for parent in parents
    ]

    return origin_node, origin_edges


def create_all_nodes_data(parents: list, children: list) -> list:
    YELLOW = "#b5de2b"
    GREEN_TOPIC = "#21918c"
    PURPLE = "#2B2E57"

    nodes = []

    nodes.extend(create_node_parent_data(parents, node_size=30, fontsize=30))
    nodes.extend(create_node_data(children, color="black", node_size=30, fontsize=30))

    return nodes


def create_network_base_data(data, summary):
    return {
        child: {
            "Parent": [{"Keyword": parent, "Relation": ""} for parent in parents],
            "Summary": summary[child],
            "Query": "",
            "Japanese_Summary": "",
            "Count": 20,
            "Japanese_Name": "",
        }
        for child, parents in data.items()
    }


def create_network_base_data_temp(children_parent):
    def transform_data(input_data: dict) -> dict:
        result_data = {}
        for key, values in input_data.items():
            for value in values:
                if value in result_data:
                    result_data[value].append(key)
                else:
                    result_data[value] = [key]
        return result_data

    parents = {
        parent: {
            "Parent": [{"Keyword": "ORIGIN", "Relation": ""}],
            "Summary": "",
            "Query": "",
            "Japanese_Summary": "",
            "Count": 20,
            "Japanese_Name": "",
        }
        for parent, children in transform_data(children_parent).items()
    }

    return parents


def main():
    children_parent, parents, children = topics()
    summary = topics_summary()
    origin_node, origin_edges = create_origin(parents)
    # papers = fetch_papers(summary)
    # with open("./data/papers-long.json", "w") as f:
    #    f.write(str(papers))

    papers = literal_eval(open("./data/papers-long.json", "r").readlines()[0])

    for key, papers in papers.items():
        print(key)
        print(len(papers))

    edges: list = create_edges_data(children_parent)
    with open("./data/edge.json", "w") as f:
        # edges.extend(origin_edges)
        # edges.extend(create_child_relation_edges())
        json.dump(edges, f)

    nodes: list = create_all_nodes_data(parents, children)
    with open("./data/node.json", "w") as f:
        # nodes.append(origin_node)
        json.dump(nodes, f)

    data = create_network_base_data(children_parent, summary)
    with open("./data/data.json", "w") as f:
        temp_data = create_network_base_data_temp(children_parent)
        data.update(temp_data)
        json.dump(data, f)


if __name__ == "__main__":
    main()

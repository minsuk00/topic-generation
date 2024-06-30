import sys
import os
from pprint import pprint
from tqdm import tqdm
import numpy as np
from time import sleep

# Get the directory of the current file (test.py), then go up one level to 'qdrant'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add 'qdrant' directory to the sys.path
if base_dir not in sys.path:
    sys.path.append(base_dir)

from fact_check import FACT_CHECK, Papers
import json


class TOPIC_DATA:
    def __init__(self, topic: str, summary: str, parents: list, query: str):
        self.topic = topic.replace("/", " ")
        self.summary = summary
        self.parents = parents
        self.query = query

    def get_data(self):
        return {
            "Topic": self.topic,
            "Summary": self.summary,
            "Parent": self.parents,
            "Query": self.query,
        }


class TOPIC_DATA_LIST:
    def __init__(self, topics_list: list):
        self.topics_list = topics_list
        self.data = list[TOPIC_DATA](self.get_topics_list())

    def get_topics_list(self):
        return [
            TOPIC_DATA(
                topic=topic,
                summary=data["Summary"],
                parents=data["Parent"],
                query=data["Query"],
            )
            for topic, data in self.topics_list.items()
        ]

    def get_data(self) -> list:
        return [item.get_data() for item in self.data]


class BUILD_TOPIC_DATA(FACT_CHECK):
    def __init__(self, topic: str, description: str) -> None:
        self.topic = topic
        self.description = description
        self.general_topics: TOPIC_DATA_LIST = self.get_general_topics()
        self.main_topics: dict[str, TOPIC_DATA_LIST] = self.get_main_topics()
        self.sub_topics: dict[str, TOPIC_DATA_LIST] = self.get_sub_topics()
        self.vectors = {}
        super().__init__(save=False, n_result=4000)

    def get_general_topics(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = f"{base_dir}/{self.topic}/General Topics Response.json"
        if not os.path.isfile(data_path):
            return []

        with open(data_path, "r+") as f:
            raw_data = dict(json.load(f))
            data = TOPIC_DATA_LIST(raw_data)
            f.close()
            return data

    def get_sub_topic(self, topic: str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = f"{base_dir}/{self.topic}/topic/{topic} - sub.json"
        if not os.path.isfile(data_path):
            return []

        with open(data_path, "r+") as f:
            raw_data = dict(json.load(f))
            data = TOPIC_DATA_LIST(raw_data)
            f.close()
            return data

    def get_main_topic(self, topic: str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = f"{base_dir}/{self.topic}/topic/{topic}.json"
        if not os.path.isfile(data_path):
            return []

        with open(data_path, "r+") as f:
            raw_data = dict(json.load(f))
            data = TOPIC_DATA_LIST(raw_data)
            f.close()
            return data

    def get_main_topics(self):
        return {
            item.topic: self.get_main_topic(topic=item.topic)
            for item in self.general_topics.data
        }

    def get_sub_topics(self):

        return {
            item.topic: self.get_sub_topic(topic=item.topic)
            for item in self.general_topics.data
        }

    def save_fact_check_data(self, data: dict, directory: str, filename: str):
        dir = f"{base_dir}/test/{self.topic}/output/{directory}"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/{filename}.json", "w+") as f:
            json.dump(data, f)
            f.close()

    def fact_check_topics(
        self,
        topics_data: TOPIC_DATA_LIST,
        directory: str,
        filename: str,
        description: str,
    ):
        data = {}
        vectors = {}
        vector_dir = f"{base_dir}/test/{self.topic}/output/vector"
        vector_file = f"{vector_dir}/{filename}.npz"
        paper_dir = f"{base_dir}/test/{self.topic}/output/{directory}"
        paper_file = f"{paper_dir}/{filename}.json"
        if os.path.isfile(vector_file) and os.path.isfile(paper_file):
            # tqdm.write(f"{item.topic} -- READ")
            return json.load(open(paper_file))

        os.makedirs(vector_dir, exist_ok=True)
        os.makedirs(paper_dir, exist_ok=True)

        for item in tqdm(
            topics_data.data, desc=f"Fact Check Data ({description})", leave=False
        ):

            query_string = f"{item.topic} {item.query} {item.summary}"
            self.check_query(query=query_string, with_vector=True)
            papers_result = self.papers_result.get_data()
            for topic_data in papers_result:
                del topic_data["id"]
                del topic_data["concepts"]

            data[item.topic] = papers_result
            vectors[item.topic] = self.papers_result.get_vector()
            self.vectors[item.topic] = self.papers_result.get_vector()
            np.savez(
                vector_file,
                "w+",
                **vectors,
            )
            self.save_fact_check_data(
                data=data,
                directory=directory,
                filename=filename,
            )
            tqdm.write(f"{item.topic} -- SAVE")

        sleep(3.5)
        return data

    def fact_check_general_topics(self):
        dir = "fact_check"
        filename = "general_topics"
        return self.fact_check_topics(
            topics_data=self.general_topics,
            directory=dir,
            filename=filename,
            description="General Topics",
        )

    def fact_check_main_topics(self):
        dir = "fact_check"
        data = {}
        for key, topic_list in self.main_topics.items():
            _dir = f"{dir}"
            filename = f"{key}"
            data[key] = self.fact_check_topics(
                topics_data=topic_list,
                directory=_dir,
                filename=filename,
                description="Main Topics",
            )
        return data

    def fact_check_sub_topics(self):
        dir = "fact_check"
        data = {}
        for key, topic_list in self.sub_topics.items():
            _dir = f"{dir}"
            filename = f"{key} - sub"
            data[key] = self.fact_check_topics(
                topics_data=topic_list,
                directory=_dir,
                filename=filename,
                description="Sub Topics",
            )
        return data

    def convert_to_useful(self, topic: str, papers: dict):
        dir = f"{base_dir}/test/{self.topic}/output/showpapers"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/{topic}.json", "w+") as f:
            for key, papers_data in papers.items():
                for paper in papers_data:
                    if not len(paper["authors"]):
                        continue
                    first_author = paper["authors"][0]
                    paper["author"] = {
                        "author_position": first_author["author"],
                        "author": {
                            "display_name": first_author["author"],
                        },
                        "institutions": [
                            {
                                "display_name": first_author["affiliation"],
                            },
                        ],
                    }
                    paper["abstract"] = " ".join(paper["abstract"].split()[:30])
                    del paper["concepts"]
                    del paper["authors"]

            json.dump(papers, f)

    def fact_check_all_topics(self):

        # self.check_query(query=f"{self.topic} {self.description}", with_vector=True)
        # base_papers = self.papers_result.get_data()
        # self.vectors[self.topic] = self.papers_result.get_vector()

        genera_topics_papers = self.fact_check_general_topics()
        main_topics_papers = self.fact_check_main_topics()
        sub_topics = self.fact_check_sub_topics()
        dir = f"{base_dir}/test/{self.topic}/vectors"
        os.makedirs(dir, exist_ok=True)
        for general_topic in self.general_topics.data:
            papers = {
                # self.topic: base_papers,
                # general_topic.topic: genera_topics_papers[general_topic.topic],
            }
            papers.update(main_topics_papers[general_topic.topic])
            papers.update(sub_topics[general_topic.topic])
            self.convert_to_useful(topic=general_topic, papers=papers)

            vectors = {key: self.vectors[key] for key in papers.keys()}
            query_target = (
                f"{general_topic.topic} {general_topic.query} {general_topic.summary}"
            )
            topics = {
                general_topic.topic: {
                    "target_vector": self.encode_text(query_target),
                    "vectors": self.vectors[general_topic.topic],
                },
                self.topic: {
                    "target_vector": self.encode_text(
                        f"{self.topic} {self.description}"
                    ),
                    "vectors": self.vectors[self.topic],
                },
            }
            topics.update(
                {
                    main_topic.topic: {
                        "target_vector": self.encode_text(
                            f"{main_topic.topic} {main_topic.summary}"
                        ),
                        "vectors": vectors[main_topic.topic],
                    }
                    for main_topic in self.main_topics[general_topic.topic].data
                }
            )

            """self.create_vector_map(
                target_vector=self.encode_text(query_target),
                vectors=self.vectors[general_topic.topic],
            )"""
            # self.create_vector_map_many(topics=topics)

            np.savez(f"{dir}/{general_topic.topic}.npz", **vectors)

    def get(self):
        genera_topics_papers = self.fact_check_general_topics()
        # self.fact_check_all_topics()

    def create_vector_map(self, target_vector, vectors):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import matplotlib.colors as mcolors

        vectors = np.array(vectors)
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
        vector = scaler.transform(np.array(target_vector).reshape(1, -1))
        vectors = np.vstack([vectors, vector])

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(vectors)

        top_100_vectors = tsne_results[:100]
        reduced_vectors = tsne_results[100:-1]
        reduced_query = tsne_results[-1]

        colors = np.linspace(0, 1, len(reduced_vectors))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "gradient", ["yellow", "orange"]
        )  # Gradient from blue to orange
        norm = mcolors.Normalize(vmin=0, vmax=1)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            reduced_vectors[:, 0],
            reduced_vectors[:, 1],
            c=colors,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
            label="Data Vectors",
        )
        plt.scatter(
            top_100_vectors[:, 0],
            top_100_vectors[:, 1],
            color="red",
            label="Top 100 Vectors",
        )
        plt.scatter(
            reduced_query[0],
            reduced_query[1],
            color="black",
            label="Query Vector",
            edgecolor="k",
            s=100,
        )
        plt.title("Visualization of Vectors")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm), orientation="vertical"
        )
        plt.legend()
        plt.show()

    def create_vector_map_many_2(self, topics):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import mplcursors

        # Initialize the scaler and TSNE
        scaler = StandardScaler()
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        plt.figure(figsize=(16, 12))

        # Preparing the dataset
        all_vectors = []  # To hold all vectors and target vectors for TSNE
        all_labels = []  # To track the topic of each vector for coloring
        vector_counts = []  # To keep track of how many vectors belong to each topic

        # Concatenate vectors from all topics
        for topic_data in topics.values():
            vectors = np.array(topic_data["vectors"])
            all_vectors.append(vectors)
            vector_counts.append(len(vectors))

        # Concatenate target vectors at the end
        for topic_data in topics.values():
            target_vector = np.array(topic_data["target_vector"]).reshape(1, -1)
            all_vectors.append(target_vector)

        # Flatten the list of vectors
        all_vectors = np.vstack(all_vectors)

        # Scale the vectors
        all_vectors_scaled = scaler.fit_transform(all_vectors)

        print("TRANSFORM")
        # Apply TSNE
        tsne_results = tsne.fit_transform(all_vectors_scaled)
        print("TRANSFORM -- DONE")
        # Generate a color map for each topic
        color_map = plt.cm.rainbow(np.linspace(0, 1, len(topics)))

        # Plotting
        start_idx = 0
        for i, (topic_name, topic_data) in enumerate(topics.items()):
            num_vectors = vector_counts[i]
            # Indices for the current topic's vectors
            indices = range(start_idx, start_idx + num_vectors)
            # Indices for the top 100 vectors, if the topic has that many
            top_indices = sorted(
                indices,
                key=lambda idx: np.linalg.norm(
                    tsne_results[idx] - tsne_results[-len(topics) + i]
                ),
                reverse=True,
            )[:20]

            # Plot all vectors for the topic with more transparency
            """plt.scatter(
                tsne_results[indices, 0],
                tsne_results[indices, 1],
                color=color_map[i],
                alpha=0.075,
            )"""

            # Highlight the top 100 vectors for the topic
            """plt.scatter(
                tsne_results[top_indices, 0],
                tsne_results[top_indices, 1],
                color=color_map[i],
                alpha=0.5,
            )"""

            # Plot the target vector for the topic
            """scatter = plt.scatter(
                tsne_results[start_idx + num_vectors, 0],
                tsne_results[start_idx + num_vectors, 1],
                color=color_map[i],
                edgecolor="k",
                marker="*",
                s=200,
                # label=f"{topic_name}",
            )"""

            # Annotate the target vector with the topic name
            plt.text(
                tsne_results[start_idx + num_vectors, 0] - 0.05,
                tsne_results[start_idx + num_vectors, 1],
                topic_name,
                fontsize=8,
                ha="right",
                va="bottom",
            )

            start_idx += num_vectors + 1  # Move to the next block of vectors

        # plt.title("Visualization of Vectors by Topic")
        # plt.xlabel("t-SNE Component 1")
        # plt.ylabel("t-SNE Component 2")
        plt.tight_layout()
        plt.axis("off")
        # plt.legend()
        plt.show()

    def create_vector_map_many(self, topics):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import matplotlib.colors as mcolors

        # Initialize TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        scaler = StandardScaler()

        plt.figure(figsize=(16, 12))

        # Generate a color map for each topic
        base_colors = plt.cm.rainbow(np.linspace(0, 1, len(topics)))
        colors = iter(base_colors)

        for topic_name, topic_data in tqdm(topics.items(), desc="Transform"):
            target_vector = topic_data["target_vector"]
            vectors = np.array(topic_data["vectors"])

            # Scale the vectors
            vectors = scaler.fit_transform(vectors)
            target_vector = scaler.transform(np.array(target_vector).reshape(1, -1))
            vectors = np.vstack([vectors, target_vector])

            # Apply TSNE
            tsne_results = tsne.fit_transform(vectors)

            # Separate the last entry as the target vector
            reduced_vectors = tsne_results[:-1]
            reduced_query = tsne_results[-1]

            # Assign the next color in the color map to this topic
            base_color = next(colors)
            emphasized_color = base_color * np.array(
                [1, 1, 1, 1.5]
            )  # Brighten the base color for emphasis

            # Plot all vectors with a more transparent color
            plt.scatter(
                reduced_vectors[100:, 0],  # Skip top 100 vectors
                reduced_vectors[100:, 1],
                color=base_color,
                alpha=0.1,  # More transparent
            )

            # Emphasize top 100 vectors with a stronger color
            if len(reduced_vectors) > 100:
                plt.scatter(
                    reduced_vectors[:100, 0],
                    reduced_vectors[:100, 1],
                    color=base_color,  # Use emphasized color for top 100 vectors
                    alpha=1.0,
                    # Less transparency for emphasis
                    s=50,
                    label=f"Top 100 {topic_name} Vectors",
                )

            # Plot the target vector with an emphasized color and bigger size
            plt.scatter(
                reduced_query[0],
                reduced_query[1],
                color=base_color,  # Use emphasized color for the target vector
                edgecolor="k",
                s=100,  # Bigger size for visibility
                label=f"{topic_name} Target Vector",
            )

        plt.title("Visualization of Vectors by Topic")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.show()

    def create_vector_map_many_interactive(self, topics):

        import plotly.graph_objects as go
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        from tqdm import tqdm
        import plotly.express as px
        import matplotlib.pyplot as plt
        import pandas as pd
        import textwrap

        scaler = StandardScaler()

        # Collecting all vectors and target vectors
        all_vectors = []
        all_labels = []
        target_indices = []
        for topic_name, topic_data in tqdm(
            topics.items(), desc="Stack Vectors", leave=False
        ):
            vectors = np.array(topic_data["vectors"])
            target_vector = np.array(topic_data["target_vector"]).reshape(1, -1)
            all_vectors.extend(vectors)
            target_index = len(all_vectors)  # Target vector index
            all_vectors.append(target_vector[0])
            all_labels.extend([f"{topic_name} Vector"] * len(vectors))
            all_labels.append(f"{topic_name} Target Vector")
            target_indices.append(target_index)

        # Scaling the vectors
        all_vectors_scaled = scaler.fit_transform(np.array(all_vectors))

        # TSNE transform
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(all_vectors_scaled)

        fig = go.Figure()

        # Preparing the dataset
        all_vectors = []  # To hold all vectors and target vectors for TSNE
        all_labels = []  # To track the topic of each vector for coloring
        vector_counts = []  # To keep track of how many vectors belong to each topic

        # Concatenate vectors from all topics
        for topic_data in topics.values():
            vectors = np.array(topic_data["vectors"])
            all_vectors.append(vectors)
            vector_counts.append(len(vectors))

        # Concatenate target vectors at the end
        for topic_data in topics.values():
            target_vector = np.array(topic_data["target_vector"]).reshape(1, -1)
            all_vectors.append(target_vector)

        # Flatten the list of vectors
        all_vectors = np.vstack(all_vectors)

        # Scale the vectors
        all_vectors_scaled = scaler.fit_transform(all_vectors)

        tqdm.write("TRANSFORM")
        # Apply TSNE
        tsne_results = tsne.fit_transform(all_vectors_scaled)
        tqdm.write("TRANSFORM -- DONE")
        # Generate a color map for each topic
        color_map = px.colors.qualitative.Light24
        color_map.extend(px.colors.qualitative.Dark24)
        color_map.extend(px.colors.qualitative.Alphabet)

        # Plotting
        start_idx = 0
        for i, (topic_name, topic_data) in tqdm(enumerate(topics.items())):
            tqdm.write(topic_name)
            num_vectors = vector_counts[i]
            papers = topic_data["papers"]
            # Indices for the current topic's vectors
            indices = range(start_idx, start_idx + num_vectors)
            # Indices for the top 100 vectors, if the topic has that many
            top_indices = sorted(
                indices,
                key=lambda idx: np.linalg.norm(
                    tsne_results[idx] - tsne_results[-len(topics) + i]
                ),
                reverse=True,
            )
            wrap = lambda txt: "<br>".join(textwrap.wrap(txt, width=50))
            format_str = "<b>Title</b>:<br>%{customdata.title}<br><br><b>Abstract</b>:<br>%{customdata.abstract}<br><extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=tsne_results[top_indices, 0],
                    y=tsne_results[top_indices, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=color_map[i],
                    ),
                    legendgroup=topic_name,
                    name=topic_name,
                    # text=[paper["title"] for paper in papers[:100]],
                    opacity=0.5,
                    customdata=[
                        {
                            "title": wrap(paper["title"]),
                            "abstract": wrap(paper["abstract"]),
                        }
                        for paper in papers
                    ],
                    hoverinfo="text",
                    hovertemplate=format_str,
                )
            )

            """fig.add_trace(
                go.Scatter(
                    x=[tsne_results[start_idx + num_vectors, 0]],
                    y=[tsne_results[start_idx + num_vectors, 1]],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=14,
                        color=color_map[i],
                        line=dict(width=2),
                    ),
                    text=topic_name,
                    hoverinfo="text",
                    name=f"{topic_name} Target point",
                    legendgroup=topic_name,
                )
            )"""

            start_idx += num_vectors + 1  # Move to the next block of vectors

        fig.update_layout(
            title="Interactive Visualization of Vectors by Topic",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            legend_title="Legend",
            template="plotly_white",
        )

        fig.show()

    def read(self):
        dir = f"{base_dir}/test/{self.topic}/vectors"
        for general_topic in self.general_topics.data:
            data = np.load(f"{dir}/{general_topic.topic}.npz", allow_pickle=True)
            query_target = (
                f"{general_topic.topic} {general_topic.query} {general_topic.summary}"
            )
            topic = {
                key: {
                    "target_vector": self.encode_text(query_target),
                    "vectors": data[key],
                }
                for key in data.files[
                    0:3  # 2 + len(self.main_topics[general_topic.topic].data)
                ]
            }
            # self.create_vector_map_many(topic)
            self.create_vector_map_many_2(topic)
            # self.create_vector_map_many_interactive(topic)

    def read_general_topic(self):

        vector_dir = f"{base_dir}/test/{self.topic}/output/vector"
        paper_dir = f"{base_dir}/test/{self.topic}/output/fact_check"

        topic = {}
        vector_data = np.load(f"{vector_dir}/general_topics.npz", allow_pickle=True)
        paper_data = json.load(open(f"{paper_dir}/general_topics.json"))
        print(vector_data.files)
        for general_topic in tqdm(
            self.general_topics.data, desc="Store Data", leave=False
        ):
            query_target = (
                f"{general_topic.topic} {general_topic.query} {general_topic.summary}"
            )
            if general_topic.topic in vector_data.files:
                topic[general_topic.topic] = {
                    "target_vector": self.encode_text(query_target),
                    "vectors": vector_data[general_topic.topic][:1000],
                    "papers": paper_data[general_topic.topic],
                }
        # create_vector_map_many(topic)
        # self.create_vector_map_many_2(topic)
        self.create_vector_map_many_interactive(topic)


def main():

    topic_data = {
        "SAF": "Sustainable Aviation Fuel (SAF) is a green alternative to traditional jet fuels, derived from renewable resources like plant materials, waste oils, and fats, designed to reduce aviation's carbon footprint and support environmental sustainability. Recent research in SAF technology focuses on improving production efficiency, reducing costs, and expanding the types of sustainable feedstocks that can be used, showing promising advancements in bioengineering, chemical processing, and lifecycle analysis. The application of SAF in the aviation industry is already underway, with airlines incorporating SAF into their fuel mix to meet regulatory requirements and sustainability goals, demonstrating a significant reduction in greenhouse gas emissions. This innovation opens up new business opportunities in the renewable energy sector, supply chain logistics, and sustainable agriculture, encouraging investments in SAF production facilities and research. Looking ahead, the market and research prospects for SAF are optimistic, with expectations of increased adoption by the aviation sector, further technological breakthroughs in production processes, and supportive regulatory policies driving the industry towards a more sustainable future.",
        "Well-being": "Focusing on Biology, the biological functions involved in well-being encompass a complex interplay of physiological, genetic, and biochemical processes that contribute to physical and mental health. Recent research advances have highlighted the role of neurochemicals like serotonin and dopamine in mood regulation, the impact of genetics on predispositions to certain mental health conditions, and the intricate relationship between gut microbiota and the brain, termed the gut-brain axis, in influencing well-being. These discoveries have led to the development of novel therapies and interventions, such as personalized nutrition plans and probiotic supplements aimed at enhancing gut health to improve overall well-being. The potential for new business is vast, ranging from biotechnology firms focusing on genetic testing for personalized healthcare, to wellness companies developing targeted nutritional products. Future prospects in the market and research are promising, with an increasing focus on integrative approaches that combine traditional medical treatments with lifestyle and nutritional interventions to optimize well-being, and ongoing research into the genetic basis of mental health conditions offering the potential for breakthroughs in prevention and treatment strategies.",
        "Biosensor": "test",
    }
    topic = "Biosensor"
    data = BUILD_TOPIC_DATA(topic, topic_data[topic])
    # data.get()
    # data.read()
    data.read_general_topic()


if __name__ == "__main__":
    main()

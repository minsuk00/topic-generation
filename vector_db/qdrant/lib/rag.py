import dotenv

from llm import BEDROCK, OpenAI
from dbclient import DBClient

dotenv.load_dotenv()


class RAG_CLIENT(DBClient):
    GPT_3 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-0125-preview"
    CLAUDE2 = "anthropic.claude-v2"
    CLAUDE1 = "anthropic.claude-instant-v1"

    QUERY_ENHANCE_HELPER = (
        "You are a helpful assistant who is a professional academic writer."
    )
    NO_CITATION_HELPER = "You are a helpful assistant. Your goal is to assist users by providing insightful and in-depth responses to a wide range of queries, from academic topics to real-world questions."
    CITATION_HELPER = "You are a Retrieval-Augmented Generation model trained to provide detailed, accurate, and contextually relevant information. Your responses should leverage a comprehensive database of knowledge, ensuring that all answers are supported by verified information. Your goal is to assist users by providing insightful and in-depth responses to a wide range of queries, from academic topics to real-world questions."

    QUERY_ENHANCE_PROMPT_FILE = "./prompt/enhance_query.txt"
    NO_CITATION_PROMPT_FILE = "./prompt/no_citation.txt"
    WITH_CITATION_PROMPT_FILE = "./prompt/paper_citation.txt"

    ENHANCED_QUERY = "NONE"

    def __init__(self) -> None:
        self.query_enhance_prompt = open(self.QUERY_ENHANCE_PROMPT_FILE).read()
        self.no_citation_prompt = open(self.NO_CITATION_PROMPT_FILE).read()
        self.with_citation_prompt = open(self.WITH_CITATION_PROMPT_FILE).read()

        self.gpt4 = OpenAI(model=self.GPT_4)
        self.gpt3 = OpenAI(model=self.GPT_3)
        self.claude2 = BEDROCK(model=self.CLAUDE2)
        self.claude1 = BEDROCK(model=self.CLAUDE1)

        super().__init__(n_result=100)

    def enhance_query(self, query: str):
        self.query_enhance_prompt = self.query_enhance_prompt.replace("<query>", query)
        self.ENHANCED_QUERY = self.claude2.get_response(
            query=self.query_enhance_prompt,
            max_tokens=4000,
        )
        # self.ENHANCED_QUERY = self.claude1.get_response(
        #    self.query_enhance_prompt, max_tokens=1000
        # )

        self.no_citation_prompt = self.no_citation_prompt.replace(
            "<context>", self.ENHANCED_QUERY
        )

    def get_no_citation_answer(self, query: str, with_enhance=False):
        if with_enhance:
            self.enhance_query(query)

        self.no_citation_prompt = self.no_citation_prompt.replace("<query>", query)

        self.no_citation_answer = self.claude2.get_response(
            query=self.no_citation_prompt,
            max_tokens=4000,
        )
        # self.no_citation_answer = self.claude1.get_response(
        #    self.no_citation_prompt, max_tokens=3000
        # )

    def format_papers_list(
        self, papers: list, score_order: list, similarity_order: list, use_type="score"
    ):
        paper_citation_string = (
            lambda title, abstract, year, score: f"Title: {title}, Abstract: {abstract},Year: {year}, Score: {score}"
        )
        return "\n".join(
            [
                paper_citation_string(
                    paper["title"],
                    paper["abstract"],
                    paper["publication_date"],
                    paper["score"],
                )
                for paper in papers
            ]
        )

    def get_result(self, query: str, with_enhance=False):
        self.get_no_citation_answer(query=query, with_enhance=with_enhance)

        print("NO CITATION ANSWER:")
        print(self.no_citation_answer)

        papers = self.get_papers(self.no_citation_answer)
        papers_context = self.format_papers_list(papers, None, None)
        self.with_citation_prompt = self.with_citation_prompt.replace("<query>", query)
        self.with_citation_prompt = self.with_citation_prompt.replace(
            "<context>", self.no_citation_answer
        )
        self.with_citation_prompt = self.with_citation_prompt.replace(
            "<papers>", papers_context
        )

        self.citation_answer = self.claude2.get_response(
            query=self.with_citation_prompt,
            max_tokens=4000,
        )
        # self.citation_answer = self.claude2.get_response(
        #    self.with_citation_prompt, max_tokens=4000
        # )
        # return self.citation_answer


def test_rag():
    query = "What is AI"
    print("QUERY:", query)
    rag_client = RAG_CLIENT()
    rag_client.get_result(query, with_enhance=True)
    print("CITATION ANSWER")
    # print(rag_client.enhance_query_result)
    print(rag_client.citation_answer)


def main():
    test_rag()


if __name__ == "__main__":
    main()

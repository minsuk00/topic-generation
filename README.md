# Topic-Generation

Project done during internship at MemoryLab

## Description

Given a theme (technology), analyze from both the market perspective and technology perspective, possible usecases, difficulties, applicable markets, etc. \
Uses a chain of LLMs to analyze the topic.

## Getting Started

1. Clone from github

   ```console
   $ git clone https://github.com/minsuk00/topic-generation.git
   ```

2. Set API keys in `.env` file (e.g. anthropic api key, azure api key, etc)
3. Install dependencies

   ```console
   $ pip install -r requirements.txt
   ```

4. Set `MODE` in `code/main.py`
5. Configure `code/_config.json` file

6. Execute program

   ```console
    $ python main.py -T "technology name"
   ```

   or

   ```console
   $ sh _run.sh
   ```

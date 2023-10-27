import os
from langchain.agents import initialize_agent, load_tools, Tool
from rec_service.utils.secret.secret_manager import SecretManager
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper

class InternetAgent:

    def __init__(self):
        token = SecretManager.get_anthropic_token()
        os.environ['OPENAI_API_KEY'] = SecretManager.get_open_ai_token()
        llm = OpenAI(temperature=0)

        os.environ['SERPAPI_API_KEY'] = SecretManager.get_serpapi_token()

        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Intermediate Answer",
                func=search.run,
                description="useful for when you need to ask with search",
            )
        ]

        #tools = load_tools(['serpapi'], llm=llm)
        agent_type = AgentType.SELF_ASK_WITH_SEARCH
        self.agent = initialize_agent(tools=tools,llm=llm, agent=agent_type, verbose=True)

    def ask_agent(self, question) -> str:
        '''
        Will ask agent a question
        :param question:
        :return:
        '''
        if os.name == 'nt':
            os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"
        try:
            response = self.agent.run(input=question)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with Langchain agent because of {str(e)}")
        return response

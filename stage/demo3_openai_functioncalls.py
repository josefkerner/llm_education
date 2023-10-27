import json
from model.gpt4.gpt4_chat import GPT4_turbo_model

class OpenAIcalls:
    def __init__(self):
        self.llm = GPT4_turbo_model(
            cfg={
                'model_name': 'gpt-3.5-turbo-0613',
            }
        )

    @staticmethod
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        # Example dummy function hard coded to return the same weather
        # In production, this could be your backend API or an external API call
        weather_info = {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
        return json.dumps(weather_info)

    def parse_response(self, response):
        # Step 2: check if GPT wanted to call a function
        response_message = response["choices"][0]["message"]
        if response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": OpenAIcalls.get_current_weather,
            }  # only one function in this example, but you can have multiple
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )

            return response_message, function_response, function_name
        else:
            return response_message, None, None

    def test_openai_function_call(self):
        question = "What is the weather in San Francisco?"
        functions = [
        {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
        ]
        messages = [{"role": "user", "content": question}]
        # generate answers from the function call with llm
        response = self.llm.call_chatgpt(
            prompt_messages=messages,
            functions=functions
        )
        # parse the response
        response_message, function_response, function_name = self.parse_response(response)

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        answer = self.llm.call_n_parse(
            prompts=[messages],
            config=self.llm.cfg,
            functions=None

        )
        return answer

if __name__ == '__main__':
    openai = OpenAIcalls()
    openai.test_openai_function_call()

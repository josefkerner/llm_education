from model.gpt.gpt_chat import GPT_turbo_model
import pytz
from datetime import datetime
import json
import inspect

class OpenAIcalls:
    def __init__(self):
        self.llm = GPT_turbo_model(
            cfg={
                'model_name': 'gpt-3.5-turbo-0613',
            }
        )
    @staticmethod
    def check_args(function, args):
        sig = inspect.signature(function)
        params = sig.parameters

        # Check if there are extra arguments
        for name in args:
            if name not in params:
                return False
        # Check if the required arguments are provided
        for name, param in params.items():
            if param.default is param.empty and name not in args:
                return False

        return True

    @staticmethod
    def get_current_time(location):
        '''
        Get the current time in a given location
        :param location:
        :return:
        '''
        try:
            # Get the timezone for the city
            timezone = pytz.timezone(location)

            # Get the current time in the timezone
            now = datetime.now(timezone)
            current_time = now.strftime("%I:%M:%S %p")

            return current_time
        except:
            return "Sorry, I couldn't find the timezone for that location."

    def parse_response(self, response, available_functions):
        # Step 2: check if GPT wanted to call a function
        response_message = response["choices"][0]["message"]
        if response_message.get("function_call"):
            print("Recommended Function call:")
            print(response_message.get("function_call"))

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            function_name = response_message["function_call"]["name"]

            # verify function exists
            if function_name not in available_functions:
                return "Function " + function_name + " does not exist"
            function_to_call = available_functions[function_name]

            # verify function has correct number of arguments
            function_args = json.loads(response_message["function_call"]["arguments"])
            print("Function arguments:")
            print(function_args)
            if OpenAIcalls.check_args(function_to_call, function_args) is False:
                return "Invalid number of arguments for function: " + function_name
            function_response = function_to_call(**function_args)

            print("Output of function call:")
            print(function_response)

            # Step 4: send the info on the function call and function response to GPT
            return response_message, function_response, function_name


        else:
            return response_message, None, None

    def test_openai_function_call(self, question: str):
        functions = [
            {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {

                        "location": {
                            "type": "string",
                            "description": "The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London",
                        }
                    },
                    "required": ["location"], #inferred by GPT
                },
            }

        ]
        available_functions = {
            "get_current_time": OpenAIcalls.get_current_time,
            #place to add another function
        }
        messages = [
            {"role": "user", "content": question}]
        # generate answers from the function call with llm
        response = self.llm.call_chatgpt(
            prompt_messages=messages,
            functions=functions
        )
        # parse the response
        response_message, function_response, function_name = self.parse_response(
            response,
            available_functions=available_functions
        )

        # Step 4: send the info on the function call and function response to GPT

        # adding assistant response to messages
        messages.append(
            {
                "role": response_message["role"],
                "function_call": {
                    "name": response_message["function_call"]["name"],
                    "arguments": response_message["function_call"]["arguments"],
                },
                "content": None
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("Messages in second request:")
        for message in messages:
            print(message)
        second_response = self.llm.generate(
            prompts=[messages],
        )
        print("Second response:")
        print(second_response)
        return second_response

if __name__ == '__main__':
    openaiobj = OpenAIcalls()
    question = "What is the time?"
    openaiobj.test_openai_function_call(question)

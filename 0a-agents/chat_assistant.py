
import json

from IPython.display import display, HTML
import markdown

class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function
    
    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        function_name = tool_call_response.function.name
        arguments = json.loads(tool_call_response.function.arguments)

        call = self.functions[function_name]
        result = call(**arguments)

        return {
            "tool_call_id": tool_call_response.id, 
            "role": "tool",
            "name": function_name,
            "content": result,
        }


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


class ChatInterface:
    def input(self):
        question = input("You:")
        return question
    
    def display(self, message):
        print(message)

    def display_function_call(self, tool_call_response, result):
        call_html = f"""
            <details>
            <summary>Function call: <tt>{tool_call_response.function.name}({shorten(tool_call_response.function.arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{tool_call_response}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result['content']}</pre>
            </div>
            
            </details>
        """
        display(HTML(call_html))

    def display_response(self, message_response):
        response_html = markdown.markdown(message_response.content)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html))



class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.client = client
    
    def gpt(self, chat_messages):
        return self.client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=chat_messages,
            tools=self.tools.get_tools(),
        )


    def run(self):
        chat_messages = [
            {"role": "system", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == 'stop':  
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.gpt(chat_messages)

                message_response = response.choices[0].message
                chat_messages.append(message_response)

                tool_calls = message_response.tool_calls
                if not tool_calls:
                    self.chat_interface.display_response(message_response)
                    break

                for tool_call in tool_calls:
                    result = self.tools.function_call(tool_call)
                    chat_messages.append(result)
                    self.chat_interface.display_function_call(tool_call, result)
    



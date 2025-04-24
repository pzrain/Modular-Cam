from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from scripts.openai_api import send_prompt

import os

class MyLLM:
    def __init__(self):
        pass

    def ask_question(self, question):
        prompt = question
        model = "gpt-3.5-turbo-1106"
        res = send_prompt(model, prompt, temperature = 0.5)
        return res['content']

    def gen_question(self, prompt):
        recommend_list = "["
        question = '''Task Description: Extract the scenes that appear in the given text in order, and identify the transition action between adjacent scenes. You should pick the transition action from ["Zoom In", "Zoom Out", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down"].
Expected Output: Descriptions of scenes and the transition actions, in order. The output should be in the format of a python list.
Instruct for GPT: The scene description should contain rich information.
Example Input: "Start with a wide angle shot of a vast mountain range under a clear blue sky. Gradually zoom in to reveal a lone hiker standing on a peak".
Example Expected Output:["mountain in the background, clear blue sky", "Zoom In", "mountain, lone hiker on a peak"].
Input: {}.
Output:'''.format(prompt)
        return question

if __name__ == '__main__':
    my_llm = MyLLM()
    prompt = "Start with a long shot of fields and blue sky. The camera moves to the left and a house appears in the distance."
    print(my_llm.gen_question(prompt))
    print(my_llm.ask_question(my_llm.gen_question(prompt)))
    # print(llm_chain.run(question))


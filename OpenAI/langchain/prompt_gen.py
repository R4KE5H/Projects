import os
import openai,langchain
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

api_key=os.getenv("KEY",None)

llm = OpenAI(temperature=0.7,openai_api_key=api_key)


def Car_list(body_type, price_range):

    prompt_template_car = PromptTemplate(
        input_variables=["body_type", "price_range"],
        template="""List out {body_type} car names only of {price_range}. Return it as a comma seperator string"""
    )

    
    car_chain= LLMChain(llm=llm, prompt=prompt_template_car)
    res =car_chain.run(body_type=body_type, price_range=price_range)
    
    return res
    



def generate_details(car_name):
    # chain 1
    prompt_template_des = PromptTemplate(
        input_variables=["car_name"],
        template="Please let me know {car_name} description of it in paragraph"
    )
    
    chain_1 = LLMChain(llm=llm, prompt=prompt_template_des, output_key="Description_car")
    
    # chain 2
    prompt_template_fea = PromptTemplate(
        input_variables=["car_name"],
        template="List out Top features of {car_name} in bulletin-wise"
    )

    chain_2= LLMChain(llm=llm, prompt=prompt_template_fea, output_key="main_features")


    chain= SequentialChain(
        chains=[chain_1,chain_2],
        input_variables=["car_name"],
        output_variables=["Description_car","main_features"]
    )

    res = chain({"car_name":car_name})
    return res



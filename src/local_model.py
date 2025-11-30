import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain


class LocalModel:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=512,
    ):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        print("Model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded")

        self.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.0,
            stop=None,
            max_new_tokens=max_new_tokens,
            return_full_text=True,
            dtype=torch.bfloat16,
            device_map=None,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def call_model(self, system_prompt, user_prompt):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("user", "{user_prompt}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return response

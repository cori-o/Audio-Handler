from openai import OpenAI
import torch

class LLMModel():
    def __init__(self, config):
        self.config = config 

    def set_gpu(self, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"    
        model.to(self.device)
    
    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

class LLMOpenAI(LLMModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def set_generation_config(self):
        self.gen_config = {
            "max_tokens": self.config['max_tokens'],
            "temperature": self.config['temperature']
        }

    def set_stock_guideline(self):
        '''
        증권 종목 분석 여부 파악에 사용할 역할들을 정의합니다. 
        가이드라인은 샘플 데이터 및 분류 결과 데이터를 보고 작성하였습니다. 
        '''
        self.system_role = """
        너는 생성된 회의록을 검토하는 역할을 수행하는 전문가야. 생성된 회의록 결과를 보고, 맥락이 이상한 부분은 제외해줘
        틀린 맞춤법이 있으면 교정해줘. 이 역할만 하고, 다른건 하지마.
        """
        
    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 사용자 질문에 대해 간단 명료하게 답을 해주면 돼", sub_role="", model='gpt-4o'):
        try:
            sub_role = sub_role
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "system", "content": sub_role},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.gen_config['max_tokens'],
                temperature=self.gen_config['temperature'],
            )
        except Exception as e:
            return f"Error: {str(e)}"
        return response.choices[0].message.content
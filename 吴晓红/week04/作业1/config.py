REGEX_RULE = {
    "FilmTele-Play": ["播放", "电视剧"], # 句子是不是包含特定的单词，做出分类
    "HomeAppliance-Control": ["空调", "广播"]
}




CATEGORY_NAME = [
    'Education-Query', 'Entertainment-Query', 'Finance-Query', 'Food-Query', 'Health-Query', 'Social-Query',
     'Sports-Query', 'Study-Query', 'Work-Query', 'shopping-Query'
]

TFIDF_MODEL_PKL_PATH = "assets/weights/tfidf_ml.pkl"

BERT_MODEL_PKL_PATH = "assets/weights/bert.pt"
BERT_MODEL_PERTRAINED_PATH = "assets/models/bert-base-chinese/"

LLM_OPENAI_SERVER_URL = f"https://dashscope.aliyuncs.com/compatible-mode/v1" # ollama
LLM_OPENAI_API_KEY = "sk-1b0891fe1ab844d98139f95bdd6b402b"
LLM_MODEL_NAME = "qwen-plus"

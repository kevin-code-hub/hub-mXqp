import argparse
import json
import os
import re
from pathlib import Path

import openai


ROOT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = ROOT_DIR / 'prompts'
DATA_DIR = ROOT_DIR / 'data'

TEST_MODEL_NAME = ''
TEST_API_KEY = ''
TEST_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
TEST_TEXT = '糖醋鲤鱼怎么做啊？'


def load_label_file(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8') as fp:
        return [line.strip() for line in fp if line.strip()]


def load_examples(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as fp:
        return json.load(fp)


def format_examples(examples: list[dict]) -> str:
    lines = []
    for idx, example in enumerate(examples, start=1):
        lines.append(f'示例{idx}')
        lines.append(f'输入: {example["text"]}')
        lines.append('输出:')
        lines.append(json.dumps({
            'domain': example['domain'],
            'intent': example['intent'],
            'slots': example['slots'],
        }, ensure_ascii=False, indent=2))
    return '\n'.join(lines)


def build_system_prompt(domains: list[str], intents: list[str], slots: list[str], examples: list[dict]) -> str:
    template_path = PROMPT_DIR / 'system_prompt.txt'
    with template_path.open('r', encoding='utf-8') as fp:
        template = fp.read()
    return (
        template
        .replace('{{domains}}', ' / '.join(domains))
        .replace('{{intents}}', ' / '.join(intents))
        .replace('{{slots}}', ' / '.join(slots))
        .replace('{{few_shot_examples}}', format_examples(examples))
    )


def extract_json_object(text: str) -> dict:
    fenced_match = re.search(r'```json\s*(\{.*?\})\s*```', text, flags=re.S)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    brace_match = re.search(r'(\{.*\})', text, flags=re.S)
    if brace_match:
        return json.loads(brace_match.group(1))

    raise ValueError('模型输出中未找到有效的JSON对象。')


def clean_slots(text: str, raw_slots, allowed_slots: set[str]) -> dict:
    if not isinstance(raw_slots, dict):
        return {}

    cleaned = {}
    for slot_name, slot_value in raw_slots.items():
        if slot_name not in allowed_slots:
            continue
        if not isinstance(slot_value, str):
            continue
        slot_value = slot_value.strip()
        if not slot_value:
            continue
        if slot_value not in text:
            continue
        cleaned[slot_name] = slot_value
    return cleaned


def validate_result(text: str, result: dict, domains: set[str], intents: set[str], slots: set[str]) -> dict:
    if not isinstance(result, dict):
        raise ValueError('模型输出的JSON对象必须是一个字典。')

    domain = result.get('domain')
    if domain not in domains:
        domain = None

    intent = result.get('intent')
    if intent not in intents:
        intent = None

    cleaned_slots = clean_slots(text, result.get('slots', {}), slots)
    return {
        'domain': domain,
        'intent': intent,
        'slots': cleaned_slots,
    }


class PromptBasedExtractor:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.domains = load_label_file(DATA_DIR / 'domains.txt')
        self.intents = load_label_file(DATA_DIR / 'intents.txt')
        self.slots = load_label_file(DATA_DIR / 'slots.txt')
        self.examples = load_examples(PROMPT_DIR / 'few_shot_examples.json')
        self.system_prompt = build_system_prompt(self.domains, self.intents, self.slots, self.examples)

    def extract(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or '{}'
        parsed = extract_json_object(content)
        return validate_result(text, parsed, set(self.domains), set(self.intents), set(self.slots))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prompt-based NLU extractor for domain, intent and slots.')
    parser.add_argument('text', nargs='?', default='', help='User utterance to parse.')
    parser.add_argument('--model', default='', help='Chat model name.')
    parser.add_argument(
        '--base-url',
        default='',
        help='Compatible OpenAI base URL.',
    )
    parser.add_argument(
        '--api-key',
        default='',
        help='API key. Defaults to DASHSCOPE_API_KEY or OPENAI_API_KEY.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model or TEST_MODEL_NAME or os.getenv('DASHSCOPE_MODEL', 'qwen-plus')
    api_key = args.api_key or TEST_API_KEY or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
    base_url = args.base_url or TEST_BASE_URL or os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    text = args.text or TEST_TEXT

    if not api_key:
        raise ValueError('Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.')

    if not text:
        raise ValueError('Missing input text. Pass text on the command line or set TEST_TEXT in llm03.py.')

    extractor = PromptBasedExtractor(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    result = extractor.extract(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
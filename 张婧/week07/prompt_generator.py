class IntentSlotPromptGenerator:
    def __init__(self, domains_file, intents_file, slots_file):
        self.domains = self._load_file(domains_file)
        self.intents = self._load_file(intents_file)
        self.slots = self._load_file(slots_file)

    def _load_file(self, filepath):
        """加载文本文件中的列表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def generate_system_prompt(self):
        """生成系统提示词"""
        domains_str = '\n'.join([f"  - {d}" for d in self.domains])
        intents_str = '\n'.join([f"  - {i}" for i in self.intents])
        slots_str = '\n'.join([f"  - {s}" for s in self.slots[:20]]) + "\n  ...等60种实体类型"

        return SYSTEM_PROMPT.format(
            domains=domains_str,
            intents=intents_str,
            slots=slots_str
        )

    def create_few_shot_examples(self, train_data, num_examples=10):
        """从训练数据中创建few-shot示例"""
        examples = []
        for item in train_data[:num_examples]:
            example = {
                "user": item["text"],
                "assistant": {
                    "domain": item["domain"],
                    "intent": item["intent"],
                    "slots": item["slots"]
                }
            }
            examples.append(example)
        return examples

    def create_chat_messages(self, user_input, examples=None):
        """创建完整的对话消息"""
        messages = [
            {"role": "system", "content": self.generate_system_prompt()}
        ]

        # 添加few-shot示例
        if examples:
            for ex in examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": json.dumps(ex["assistant"], ensure_ascii=False)})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

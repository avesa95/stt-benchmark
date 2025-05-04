def format_prompt(incorrect_sentence: str) -> str:
    few_shot_examples = [
        {
            "input": "He go to store every day and buy stuffs he need.",
            "output": "He goes to the store every day and buys the things he needs.",
        },
        {
            "input": "Umm I guess maybe we should like leave now or something.",
            "output": "I guess we should leave now.",
        },
        {
            "input": "She don’t know what she wanna do after graduation.",
            "output": "She doesn't know what she wants to do after graduation.",
        },
    ]

    instructions = (
        "You are a grammar correction assistant.\n"
        "Your task is to:\n"
        "- Fix grammar and punctuation\n"
        "- Remove filler words (e.g. 'umm', 'like', 'you know')\n"
        "- Split long run-on sentences if needed\n"
        "- Make the sentence clear and concise\n\n"
        "Important:\n"
        "- Return ONLY the corrected sentence.\n"
        "- Do NOT include thoughts, reasoning, explanations, or internal monologue.\n"
        "- Do NOT repeat the instructions.\n"
        "- Keep the language the same as the input.\n"
    )

    # Add few-shot examples
    examples_text = ""
    for ex in few_shot_examples:
        examples_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n---\n"

    # Final prompt
    prompt = f"{instructions}\n{examples_text}Input: {incorrect_sentence}\nOutput:"
    return prompt


if __name__ == "__main__":
    from src.slm.models import QwenModel

    incorrect_example = (
        "So I was kinda walking to the store and then I seen this guy who I think he was in my math class but I'm not kinda super sure and then he just kind of looked at me and I didn’t really know what to do so I just kept walking, and umm then I remembered I forgot my wallet so I had to turn back.",
    )

    model = QwenModel(temperature=0.1, model_name="qwen:4b")
    results = []

    prompt = format_prompt(incorrect_example)
    output = model.complete(prompt)
    print(output)

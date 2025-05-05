def english_grammar_correction_prompt(incorrect_sentence: str) -> str:
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
        {
            "input": "So like yesterday we was talking about going to the movies but then umm I think she said she can't cause she got work or something like that I don't know.",
            "output": "Yesterday, we were talking about going to the movies, but then she said she couldn’t because she had work or something like that.",
        },
        {
            "input": "I been thinking like maybe we could go there tomorrow if the weather’s okay you know and like not too cold or rainy or anything.",
            "output": "I've been thinking maybe we could go there tomorrow if the weather's okay and not too cold or rainy.",
        },
    ]

    instructions = (
        "You are a grammar correction assistant.\n\n"
        "Your task is to correct **spoken, informal English** by making the **fewest possible changes**.\n"
        "Keep the original meaning, tone, and conversational style.\n\n"
        "Do the following:\n"
        "✅ Fix grammar, spelling, and punctuation\n"
        "✅ Remove filler/disfluency words (e.g., 'umm', 'like', 'you know') **only if they reduce clarity**\n"
        "✅ Split long sentences **only when necessary** for readability\n\n"
        "Avoid the following:\n"
        "❌ Do NOT paraphrase or reword unnecessarily\n"
        "❌ Do NOT change the tone or intention\n"
        "❌ Do NOT add or remove ideas\n"
        "❌ Do NOT say 'The corrected sentence is:' or explain anything\n\n"
        "Respond with **only the corrected sentence**.\n"
        "Maintain the same language as the input.\n"
    )

    examples_text = ""
    for ex in few_shot_examples:
        examples_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n---\n"

    return f"{instructions}\n{examples_text}Input: {incorrect_sentence}\nOutput:"

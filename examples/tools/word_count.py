"""
Example custom tool for threadwatch.

Drop this file (or any .py following this pattern) into:
  ~/.threadwatch/tools/

threadwatch will pick it up automatically on next run.

Requirements:
  - A DEFINITION dict in OpenAI function-calling format
  - An async (or sync) function whose name matches DEFINITION["function"]["name"]
  - The function receives keyword arguments matching the schema's properties
"""

DEFINITION = {
    "type": "function",
    "function": {
        "name": "word_count",
        "description": "Count the number of words, sentences, and paragraphs in a block of text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyse",
                }
            },
            "required": ["text"],
        },
    },
}


async def word_count(text: str) -> str:
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    return f"words: {words}, sentences: {sentences}, paragraphs: {paragraphs}"

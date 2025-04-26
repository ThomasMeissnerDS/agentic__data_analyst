import re

PATTERN = re.compile(r"(.*?)```tool_code\s*(.*?)\s*```(.*)", re.DOTALL)


def _txt(text: str):
    """
    Convert a string to a list of dictionaries.
    """
    return [{"type": "text", "text": text}]


def strip_string_quotes(raw_text: str) -> str:
    """
    Strip quotes from a string.
    """
    text = raw_text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text.replace("\\n", "\n")


def _shorten(text: str, limit: int) -> str:
    """
    Shorten a string to a given limit.
    """
    if len(text) <= limit:
        return text
    head, tail = text[:int(limit/2)].rstrip(), text[-int(limit/2):].lstrip()
    return f"{head}\n\n…[truncated {len(text)-limit} chars]…\n\n{tail}"


def extract_text_and_code(model_text: str):
    """Return (pre‑code text, code block, post‑code text)."""
    m = PATTERN.match(model_text.strip())
    return m.groups() if m else (model_text.strip(), None, "")


def summarize_conversation(
        log: list, 
        max_total_chars: int = 1000, 
        max_item_chars: int = 200, 
        keep_last_n_items: int = 3):
    """
    Summarize a conversation.

    :param log: list of tuples (timestamp, text)
    :param max_total_chars: maximum number of characters in the summary
    :param max_item_chars: maximum number of characters in each item
    :param keep_last_n_items: number of items to keep from the end
    """
    verbatim = [(t, _shorten(c, max_item_chars)) for t, c in log[-keep_last_n_items:]]
    if len(log) > keep_last_n_items:
        older_snips = " • ".join(
            f"[{t}] {c[:80].replace(chr(10), ' ')}…" for t, c in log[:-keep_last_n_items]
        )
        trimmed = [("SUMMARY", older_snips)] + verbatim
    else:
        trimmed = verbatim
    convo = "\n".join(f"[{t}] {c}" for t, c in trimmed)
    return trimmed, convo[-max_total_chars:]

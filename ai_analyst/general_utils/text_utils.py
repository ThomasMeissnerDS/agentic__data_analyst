import re
from ai_analyst.utils.inference import analyst_inference
from ai_analyst.analysis_kit.config import AnalysisConfig

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


def _summarize_with_llm(conversation_log: list, config: AnalysisConfig) -> str:
    """
    Use an LLM to summarize the conversation, focusing on key insights and analysis steps.
    
    Args:
        conversation_log: List of tuples (kind, content) representing the conversation
        config: Configuration object containing model settings
        
    Returns:
        str: A concise summary of the conversation
    """
    # Format the conversation for the LLM
    formatted_conversation = []
    for kind, content in conversation_log:
        if kind == "LLM":
            formatted_conversation.append(f"Analyst: {content}")
        elif kind == "TOOL":
            formatted_conversation.append(f"Tool Output: {content}")
        elif kind == "TOOL_CODE":
            formatted_conversation.append(f"Tool Code: {content}")
    
    conversation_text = "\n".join(formatted_conversation)
    
    # Create the prompt for summarization
    prompt = f"""Please provide a concise summary of the following data analysis conversation. 
Focus on:
1. Key insights discovered
2. Important analysis steps taken
3. Any patterns or relationships identified
4. Next steps or recommendations

Conversation:
{conversation_text}

Summary:"""
    
    # Use the existing inference function to get the summary
    messages = [{"role": "user", "content": prompt}]
    summary = analyst_inference(messages, config=config)
    return summary.strip()


def summarize_conversation(
        log: list, 
        max_total_chars: int = 1000, 
        max_item_chars: int = 200, 
        keep_last_n_items: int = 3,
        config: AnalysisConfig = None):
    """
    Summarize a conversation using an LLM.

    :param log: list of tuples (timestamp, text)
    :param max_total_chars: maximum number of characters in the summary
    :param max_item_chars: maximum number of characters in each item
    :param keep_last_n_items: number of items to keep from the end
    :param config: Configuration object for LLM settings
    """
    if config is None:
        config = AnalysisConfig()
    
    # Get LLM summary
    llm_summary = _summarize_with_llm(log, config)
    
    # Keep the last few items verbatim for immediate context
    verbatim = [(t, _shorten(c, max_item_chars)) for t, c in log[-keep_last_n_items:]]
    
    # Combine LLM summary with recent items
    combined = [("SUMMARY", llm_summary)] + verbatim
    convo = "\n".join(f"[{t}] {c}" for t, c in combined)
    
    return combined, convo[-max_total_chars:]

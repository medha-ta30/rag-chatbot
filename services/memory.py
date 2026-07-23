def get_recent_conversation_context(messages: list[dict], max_turns: int = 3) -> tuple[str, int]:
    """
    Extracts the most recent conversation turns from a list of message dictionaries
    and formats them into a plain-text context string for query expansion.

    Args:
        messages (list[dict]): A list of message dictionaries formatted as
            [{"role": "user" | "assistant", "content": "..."}]
        max_turns (int, optional): The maximum number of recent user-assistant turns to include. Defaults to 3.

    Returns:
        tuple[str, int]:
            - formatted_context (str): Plain-text representation of recent conversation history.
            - turn_count (int): Number of user turns included in the formatted context.
    """
    if not messages:
        return "", 0

    if max_turns <= 0:
        return "", 0

    # Retrieve at most (max_turns * 2) messages from the end of the history
    recent_messages = messages[-(max_turns * 2):]

    formatted_lines = []
    turns_count = 0

    for msg in recent_messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()

        if not content:
            continue

        role_label = "User" if role == "user" else "Assistant"
        formatted_lines.append(f"{role_label}: {content}")

        if role == "user":
            turns_count += 1

    formatted_context = "\n".join(formatted_lines)
    return formatted_context, turns_count

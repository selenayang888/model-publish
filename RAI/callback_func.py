async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
) -> dict:

    query = messages["messages"][0]["content"]
    context = None

    if "file_content" in messages["template_parameters"]:
        query += messages["template_parameters"]["file_content"]

    target = ModelEndpoints()

    response = target(query)["response"]

    # Format responses in OpenAI message protocol
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {},
    }

    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
    }

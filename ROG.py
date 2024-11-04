# Function to generate a system message in the format required by the API
def generate_system_message(system_message):
    """
    Generates a system message for the conversation.

    Args:
        system_message (str): The content of the system message.

    Returns:
        dict: A dictionary representing the system message with role and content.
    """
    return {
        "role": "system",
        "content": system_message
    }

# Function to generate a user message in the format required by the API
def generate_user_message(user_message):
    """
    Generates a user message for the conversation.

    Args:
        user_message (str): The content of the user message.

    Returns:
        dict: A dictionary representing the user message with role and content.
    """
    return {
        "role": "user",
        "content": user_message
    }

# Function to accumulate responses and manage conversation flow
def accumulate_response(current_response, accumulated_response, conversation_ongoing, conversation, stop_sequence):
    """
    Appends the current response to the accumulated response, updates conversation history,
    and checks for a stop sequence to determine if the conversation should end.

    Args:
        current_response (str): The response chunk from the API call.
        accumulated_response (str): All responses combined from previous calls.
        conversation_ongoing (bool): Indicates if the conversation is active.
        conversation (str): The conversation history to keep context.
        stop_sequence (str): The unique sequence marking the end of the generation task.

    Returns:
        tuple: Updated accumulated response, conversation status, and conversation history.
    """
    accumulated_response += current_response  # Append the current response to the accumulated result

    # Check if the stop sequence is present, signaling the end of generation
    if stop_sequence in current_response:
        conversation_ongoing = False  # Set the conversation to stop

    # Update conversation for the next prompt, adding continuation instruction
    conversation += f"\nAssistant:{current_response}\nUser:Continue immediately after where you left off."

    return accumulated_response, conversation_ongoing, conversation

# Function to call the GPT model in non-streaming mode
def call_gpt(client, model, messages, max_tokens, stop_sequence="‡‡‡‡‡"):
    """
    Calls the GPT model in non-streaming mode to generate a response, handling pagination for long outputs.

    Args:
        client: The API client instance for OpenAI.
        model (str): The model to use (e.g., "gpt-4").
        messages (list): The conversation history with system and user messages.
        max_tokens (int): The maximum tokens for each API call.
        stop_sequence (str): The unique sequence indicating end of response.

    Returns:
        str: The full accumulated response, cleaned of the stop sequence.
    """
    api_settings = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    accumulated_response = ""  # Initialize the accumulator for responses
    conversation_ongoing = True  # Flag for conversation status
    conversation = messages[-1]['content']  # Get the latest user prompt

    while conversation_ongoing:
        try:
            response = client.chat.completions.create(**api_settings)  # API call to generate response
            current_response = response.choices[0].message.content  # Extract the response content

            # Update the accumulated response and conversation status
            accumulated_response, conversation_ongoing, conversation = accumulate_response(
                current_response, accumulated_response, conversation_ongoing, conversation, stop_sequence
            )

            # Update the latest user message content to continue conversation
            messages[-1]['content'] = conversation

        except Exception as e:
            print(f"An error occurred while calling OpenAI API: {e}")  # Handle any API call errors
            raise  # Re-raise the exception for debugging

    # Remove the stop sequence from the final response
    accumulated_response = accumulated_response.replace(stop_sequence, "").strip()
    return accumulated_response.strip()

# Function to call the GPT model in streaming mode
def call_gpt_stream(client, model, messages, max_tokens, stop_sequence="‡‡‡‡‡"):
    """
    Calls the GPT model in streaming mode, yielding response chunks in real-time.

    Args:
        client: The API client instance for OpenAI.
        model (str): The model to use (e.g., "gpt-4").
        messages (list): The conversation history with system and user messages.
        max_tokens (int): The maximum tokens for each API call.
        stop_sequence (str): The unique sequence indicating end of response.

    Yields:
        str: Each chunk of content as received from the API.
    """
    api_settings = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True  # Enable streaming mode for real-time output
    }

    accumulated_response = ""  # Initialize the response accumulator
    conversation_ongoing = True  # Flag for conversation status
    conversation = messages[-1]['content']  # Get the latest user prompt

    while conversation_ongoing:
        try:
            response = client.chat.completions.create(**api_settings)  # API call to generate response in chunks
            current_response = ""  # Initialize the current response

            for chunk in response:
                value = chunk.choices[0].delta.content  # Extract content from each chunk
                if value:  # Process non-empty chunks
                    yield value  # Yield each chunk to the caller
                    current_response += value  # Append chunk to current response

            # Update accumulated response and conversation status
            accumulated_response, conversation_ongoing, conversation = accumulate_response(
                current_response, accumulated_response, conversation_ongoing, conversation, stop_sequence
            )

            # Update the latest user message content to continue conversation
            messages[-1]['content'] = conversation

        except Exception as e:
            print(f"An error occurred while calling OpenAI API: {e}")  # Handle any API call errors
            raise  # Re-raise the exception for debugging

# Function to initialize and manage the process of generating long-form responses
def generate_long_response(client, model, max_tokens, stop_sequence="‡‡‡‡‡", stream=False):
    """
    Initializes and manages the generation of a long-form response, either in streaming or non-streaming mode.

    Args:
        client: The OpenAI client instance.
        model (str): The model to use (e.g., "gpt-4").
        max_tokens (int): The maximum tokens per API call.
        stop_sequence (str): The unique sequence indicating end of response.
        stream (bool): Whether to enable streaming mode.

    Returns:
        None
    """
    # Create the initial system prompt for setting the task context
    system_prompt = f"""
    You are an AI assistant that provides detailed information on any topic.
    You will be provided with a user prompt that asks for an explanation of a complex topic, and you need to generate a detailed response.

    Input:
        1- A user prompt asking for an explanation of a complex topic.

    Output:
        1- A detailed explanation of the topic, divided into 5 main sections, and 10 subsections inside each section.
        2- Once the required output is generated to 100% completion, append the following text to the end of the response in PLAIN TEXT: '{stop_sequence}'
    """

    # Example user prompt for initiating the conversation
    user_prompt = "Explain quantum computing."

    # Create initial messages for the conversation
    messages = [
        generate_system_message(system_prompt),  # System prompt for context
        generate_user_message(user_prompt)  # User prompt for the response
    ]

    if not stream:
        # Non-streaming mode: handle full response generation
        response = call_gpt(client, model, messages, max_tokens=max_tokens, stop_sequence=stop_sequence)
        print(response)  # Print the full response

    else:
        # Streaming mode: process and print each response chunk
        double_dagger_count = 0  # Counter to track occurrences of stop sequence indicators

        for chunk in call_gpt_stream(client, model, messages, max_tokens=max_tokens, stop_sequence=stop_sequence):
            if chunk:  # Ensure only non-empty chunks are processed
                # Check for the stop sequence and handle termination
                if "‡" in chunk:
                    double_dagger_count += 1
                    if double_dagger_count == 5:  # Detect full stop sequence
                        break
                else:
                    # Print the response chunk with potential stop markers
                    print("‡" * double_dagger_count + chunk, end="")
                    double_dagger_count = 0  # Reset counter

# Main entry point to run the script
if __name__ == "__main__":
    from openai import OpenAI  # Import the OpenAI client module

    OPENAI_API_KEY = "YOUR_API_KEY"  # Replace with your OpenAI API key
    model = "gpt-4o"  # Specify the model to use

    client = OpenAI(api_key=OPENAI_API_KEY)  # Initialize the client with the API key
    max_output_tokens = 100  # Set the max tokens for output to test the ROG functionality

    # Generate long response in streaming mode
    generate_long_response(client, model, max_output_tokens, stream=True)
    print("\n\n")
    # Generate long response in non-streaming mode
    generate_long_response(client, model, max_output_tokens, stream=False)


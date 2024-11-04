# Rolling Output Generation (ROG): LLM Design Pattern for Unrestricted Output Length

## Introduction
Large Language Models (LLMs) have inherent limitations in their maximum output token length. Generating content that exceeds this limit can be challenging, as it often leads to abrupt cutoffs and incomplete results. This limitation can hinder the creation of long-form content such as detailed reports, articles, or books.

While techniques like **Retrieval-Augmented Generation (RAG)** address input context size limitations, there has been less focus on overcoming output length constraints—until now. The **Rolling Output Generation (ROG)** pattern is designed to address these output length limitations.

| Technique   | **RAG**                                                                    | **ROG**                                                                  |
| ----------- |----------------------------------------------------------------------------| ------------------------------------------------------------------------ |
| **Purpose** | Addresses **input** context limitations                                    | Addresses **output** length limitations                                  |

## What is ROG?

**ROG (Rolling Output Generation)** is an LLM design pattern for generating coherent, context-preserving long-form content from LLMs beyond their intrinsic limits, where outputs are handled in a rolling, autonomous manner.

While it is possible to generate output of any length using this approach, full coherence can only be guaranteed up to the input context length, which is still a significant improvement. For example, the latest OpenAI models have a 128K token input context length but only a 4K token output length. Using ROG, with a single initial call, it is possible to generate coherent outputs 30 times longer than the intrinsic model's output length.

## How ROG Works

An initial call starts an autonomous conversation with the model, leading it to continuously generate coherent, context-aware content until a predefined stop sequence is detected. The accumulated response is then finalized and returned.

## ROG flow

1. **Initialization**: 
   - Set up the system and user messages.
   - Initialize relevant variables.
     - **Key Point**: Ensure the system prompt instructs the model to append a unique stop sequence at the end of the **entire generation task**, not just the current segment.
2. **Invoke the Model & Accumulate Response**: 
   - Call the model.
   - Collect and append the response to the accumulated output.
3. **Prepare for Continuation**: 
   - Update the conversation history.
   - Prepare for continuation.
     - **Key Point**: Add a new user prompt instructing the model to continue from where it left off.
4. **Repeat**: 
   - Loop the process until the stopping sequence is detected.
5. **Completion**: 
   - Finalize the accumulated response.
   - Clean up the stop sequence markers.

 <div align="center">                                                                                                                                                                         
   <img src="https://github.com/user-attachments/assets/3a990cf7-0d99-47f0-9341-2c1f7cba2fdc" alt="ROG">                                                                                      
 </div>  

## Implementation

`ROG.py` contains a possible implementation of ROG using the OpenAI API. You will need an OpenAI API key to run the code. 

The code includes functions for both streaming and non-streaming modes. To showcase the implementation, we provide a simple example of generating a lengthy response from an initial prompt, while restricting the output max tokens per call to just 100 tokens.

## Applicability
**When to Use**:
- When the target output length exceeds the model's maximum token limit.
- For generating lengthy documents, books, or detailed reports.

**Scenarios**:
- Writing chapters of a book.
- Creating extensive technical documentation or comprehensive guides.

### Considerations
- **Latency**: Multiple calls may introduce delays (in non-streaming mode).
- **Resource Use**: May increase computational resources and cost.
- **Stop Sequence**: Great care must be taken to ensure the stop sequence is unique and is initiated and detected correctly. Otherwise, the output may be truncated prematurely, or generation may continue indefinitely.

## Citation 
If you find this work useful, please consider citing this repository:

```plain
Mohamed A. Haggag (2024). Rolling Output Generation (ROG): LLM Design Pattern for Unrestricted Output Length. URL: https://github.com/mo-haggag/ROG  

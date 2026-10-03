# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) chatbot built in Python using LangGraph, Pinecone, local Hugging Face embeddings, Gemini, and FastAPI.

The chatbot answers questions strictly from the provided Agentic AI eBook and refuses to answer when the requested information cannot be supported by the document.

## Architecture

```text
                         Agentic AI PDF
                               |
                               v
                        PDF Extraction
                               |
                               v
                         Text Chunking
                               |
                               v
                 Local Hugging Face Embeddings
                               |
                               v
                            Pinecone
                               |
                               v
                          User Query
                               |
                               v
                        +-------------+
                        |   LangGraph |
                        +-------------+
                               |
                               v
                         Retrieve Node
                               |
                               v
                       Relevance Check
                         /           \
                        /             \
                   Relevant       Not Relevant
                      |                 |
                      v                 v
                Generate Node      Refusal Node
                      |
                      v
                Gemini Generation
                      |
                      v
                Grounding Check
                   /        \
                  /          \
             Grounded     Not Grounded
                |               |
                v               v
          Final Answer      Refusal Node
                |               |
                +-------+-------+
                        |
                        v
                  FastAPI Response
                        |
                        v
             +--------------------------+
             | query                    |
             | final_answer             |
             | retrieved_context_chunks |
             | confidence_score         |
             +--------------------------+
          Final Answer          Refusal

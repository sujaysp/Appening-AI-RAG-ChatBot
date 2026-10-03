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
                    LangGraph
                          |
            +-------------+-------------+
            |                           |
            v                           v
        Retrieval                 Relevance Check
            |                           |
            +-------------+-------------+
                          |
                 Relevant Context
                          |
                          v
                  Gemini Generation
                          |
                          v
                   Grounding Check
                          |
                 +--------+--------+
                 |                 |
                 v                 v
              Grounded          Unsupported
                 |                 |
                 v                 v
          Final Answer          Refusal
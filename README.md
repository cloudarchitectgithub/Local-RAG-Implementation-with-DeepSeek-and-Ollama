# Local RAG Implementation with DeepSeek and Ollama

<p align="center">
<img src="https://i.imgur.com/K5hz1pV.png" height="80%" width="80%" alt="Pic"/>
<br />
<br />


## Project Description

This project implements a fully local Retrieval-Augmented Generation (RAG) system that allows users to interact with and query PDF documents through a conversational interface. The implementation leverages the power of local Large Language Models (LLMs) through Ollama, specifically using the DeepSeek model, to provide accurate and contextually relevant responses based on the content of the documents.

The RAG system operates entirely locally without requiring external API calls, making it suitable for scenarios where data privacy is paramount or internet connectivity is limited. The application extracts content from PDF documents, processes and chunks the text, creates vector embeddings, stores them in a vector database, and then uses semantic search to retrieve relevant context when responding to user queries.

The implementation features a user-friendly interface built with Chainlit, which provides a chat-like experience where users can ask questions about the document content. The system retrieves the most relevant information from the document and uses the DeepSeek LLM to generate natural language responses that are contextually accurate and helpful.

This project demonstrates a practical application of modern AI techniques for document understanding and question answering, all while maintaining data sovereignty by running everything on the local machine.

## Project Objectives

The primary objectives of this RAG implementation project are:

1. **Create a Fully Local RAG System**: Develop a complete RAG pipeline that operates entirely on the local machine without requiring external API calls or cloud services.

2. **Leverage Local LLMs**: Utilize Ollama to run the DeepSeek language model locally for generating contextually relevant responses based on document content.

3. **Implement Efficient Document Processing**: Extract, process, and chunk PDF documents effectively to prepare them for embedding and retrieval.

4. **Build a Vector Knowledge Base**: Create and maintain a vector database using Qdrant to store document embeddings for efficient semantic search.

5. **Develop a User-Friendly Interface**: Implement a conversational interface using Chainlit that allows users to interact with the system through natural language queries.

6. **Ensure Accurate Information Retrieval**: Design the system to retrieve the most relevant context from documents when responding to user queries.

7. **Demonstrate RAG Architecture Understanding**: Showcase a comprehensive understanding of the RAG process flow, including document ingestion, embedding generation, vector storage, retrieval, and response generation.

8. **Provide a Reusable Implementation**: Create a solution that can be easily adapted for different documents and use cases with minimal modifications.

## Tools and Technologies

This RAG implementation leverages several modern tools and technologies to create a fully local solution for document question-answering. Below is a detailed breakdown of the key components used in this project:

### Core Technologies

#### 1. Large Language Model (LLM)
- **Ollama**: A framework for running large language models locally
- **DeepSeek R1**: The specific LLM used in this implementation, running locally through Ollama

#### 2. Vector Database
- **Qdrant**: A vector similarity search engine that runs locally via Docker
- Used for storing and retrieving document embeddings based on semantic similarity

#### 3. Embedding Models
- **Hugging Face Embeddings**: Specifically using the "sentence-transformers/all-MiniLM-L6-v2" model
- Transforms text chunks into vector representations for semantic search

#### 4. Document Processing
- **Docling**: A Python package for extracting and processing information from PDF documents
- **HybridChunker**: Used for intelligent document chunking based on content structure

#### 5. Framework and Orchestration
- **LangChain**: Framework for developing applications powered by language models
  - Used for connecting various components of the RAG pipeline
  - Provides document loaders, text splitters, and vector store integration
- **LangChain components used**:
  - `RecursiveCharacterTextSplitter` and `MarkdownHeaderTextSplitter` for text chunking
  - `HuggingFaceEmbeddings` for generating embeddings
  - `QdrantVectorStore` for vector database integration
  - `DirectoryLoader` and `DoclingLoader` for document loading

#### 6. User Interface
- **Chainlit**: A Python package for creating chat-based interfaces for LLM applications
  - Provides a user-friendly web interface for interacting with the RAG system
  - Displays thinking steps and response generation process

### Development Tools

#### 1. Environment Management
- **UV**: A Python package installer and resolver used for dependency management
- **Python 3.10+**: The programming language used for implementation

#### 2. Containerization
- **Docker**: Used for running Qdrant vector database locally

#### 3. Version Control
- **Git**: For source code management and version control

### File Formats
- **PDF**: The primary document format processed by the application
- **Markdown**: Used for intermediate document representation and processing

### Configuration
- **Environment Variables**: Used for storing configuration settings like database URLs
- **dotenv**: Python package for loading environment variables from .env files

This combination of technologies enables a fully local RAG implementation that doesn't rely on external APIs or cloud services, providing privacy, control, and independence from internet connectivity while still leveraging the power of modern AI techniques for document understanding and question answering.

## Project Solution

This section provides a detailed, step-by-step guide to implementing the Local RAG system with DeepSeek and Ollama. The implementation follows a structured approach to create a fully functional document question-answering system that runs entirely on your local machine.

### Prerequisites

Before starting the implementation, ensure you have the following prerequisites installed:
1. **OS**: Windows (WSL2 recommended)/maxOS/Linux
2. **RAM**: 16GB+ (32GB recommended for 7B+ models)
3. **Python 3.10+**: The core programming language used for this implementation
4. **Docker**: Required for running Qdrant vector database locally
5. **UV**: A Python package installer and resolver for dependency management

### Setup Instructions

#### Step 1: Setting Up the Environment

1. **Set Up Project Directory**
   ```bash
   # Create project directory
   mkdir local-rag-project
   cd local-rag-project
   
   # Create necessary subdirectories
   mkdir -p data
   ```

2. **Install Dependencies Using UV**
   
   If you don't have UV installed, you can install it following the instructions on the official website. UV will create a virtual environment and install all necessary packages.
   
   ```bash
   uv sync
   ```

3. **Set Up Qdrant Vector Database**
   
   Qdrant is used as the vector database for storing document embeddings. Run it locally using Docker:
   
   ```bash
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -p 6334:6334 \
   -v $(pwd)/qdrant_storage:/qdrant/storage:z \
   qdrant/qdrant
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file by copying the example file:
   
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to include your Qdrant URL:
   
   ```
   QDRANT_URL_LOCALHOST="http://localhost:6333"
   ```

#### Step 2: Data Ingestion

The data ingestion process involves extracting content from PDF documents, processing the text, and preparing it for embedding and storage in the vector database.

1. **Prepare Your PDF Document**
   
   Place your PDF document in the `data` directory. The default implementation uses a file named `DeepSeek_R1.pdf`, but you can modify the `FILE_PATH` variable in `ingest.py` to point to your document.

2. **Understanding the Ingestion Process**
   
   The `ingest.py` file contains the code for processing the PDF document and creating the vector database. Here's a breakdown of the key components:

   ```python
   # Key components from ingest.py
   
   # Environment setup
   qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
   EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
   EXPORT_TYPE = ExportType.DOC_CHUNKS
   FILE_PATH = "./data/DeepSeek_R1.pdf"
   
   # Document loading and chunking
   loader = DoclingLoader(
       file_path=FILE_PATH,
       export_type=EXPORT_TYPE,
       chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
   )
   
   docling_documents = loader.load()
   
   # Embedding generation
   embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
   
   # Vector database creation
   vectorstore = QdrantVectorStore.from_documents(
       documents=splits,
       embedding=embedding,
       url=qdrant_url,
       collection_name="rag",
   )
   ```

3. **Run the Ingestion Process**
   
   Execute the ingestion script to process the document and create the vector database:
   
   ```bash
   uv run ingest.py
   ```
   
   This command will:
   - Extract content from the PDF document
   - Split the content into appropriate chunks
   - Generate embeddings for each chunk
   - Store the embeddings in the Qdrant vector database

#### Step 3: Vector Database Creation

The vector database is created during the ingestion process. Let's understand how it works:

1. **Document Chunking**
   
   The document is split into manageable chunks using the HybridChunker from Docling, which intelligently breaks down the document based on content structure.

2. **Embedding Generation**
   
   Each chunk is converted into a vector embedding using the Hugging Face embedding model "sentence-transformers/all-MiniLM-L6-v2".

3. **Vector Storage**
   
   The embeddings are stored in the Qdrant vector database with a collection name "rag". This collection will be used for semantic search when answering user queries.

#### Step 4: RAG Implementation

The RAG implementation is contained in the `rag-chainlit-deepseek.py` file, which orchestrates the process of receiving user queries, retrieving relevant context, and generating responses.

1. **Understanding the RAG Implementation**
   
   Here's a breakdown of the key components in the RAG implementation:

   ```python
   # Key components from rag-chainlit-deepseek.py
   
   # Initialize the embedding model (same as used during ingestion)
   embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
   
   # Connect to the Qdrant vector store
   vectorstore = QdrantVectorStore(
       url=qdrant_url,
       collection_name="rag",
       embeddings=embeddings,
   )
   
   # Create a retriever from the vector store
   retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
   
   # Initialize the Ollama LLM
   llm = Ollama(model="deepseek:latest")
   
   # Create a RAG chain
   rag_chain = (
       {"context": retriever, "question": RunnablePassthrough()}
       | prompt
       | llm
       | StrOutputParser()
   )
   ```

2. **Chainlit Integration**
   
   The implementation uses Chainlit to create a user-friendly chat interface. The main components include:

   ```python
   # Chainlit setup
   @cl.on_chat_start
   async def on_chat_start():
       # Initialize the session
       
   @cl.on_message
   async def on_message(message: cl.Message):
       # Process user message
       # Retrieve context
       # Generate response
   ```

#### Step 5: Running the Application

Once the ingestion process is complete and the vector database is populated, you can run the RAG application:

1. **Start the Chainlit Application**
   
   ```bash
   uv run chainlit run rag-chainlit-deepseek.py
   ```

2. **Interact with the Application**
   
   - Open your browser and navigate to the URL provided in the terminal (typically http://localhost:8000)
   - Start asking questions about the content of your PDF document
   - The system will retrieve relevant context from the vector database and generate responses using the DeepSeek LLM

3. **Understanding the Response Generation Process**
   
   When you ask a question:
   - The question is converted to an embedding using the same model used during ingestion
   - Semantic search is performed in the vector database to find the most relevant chunks
   - The retrieved chunks are combined with your question to form a prompt for the LLM
   - The DeepSeek LLM generates a response based on the provided context and question
   - The response is displayed in the Chainlit interface

### Implementation Details

#### Key Files and Their Functions

1. **ingest.py**
   - Handles document processing and vector database creation
   - Uses Docling for PDF extraction and processing
   - Generates embeddings and stores them in Qdrant

2. **rag-chainlit-deepseek.py**
   - Implements the RAG pipeline
   - Connects to the vector database
   - Initializes the Ollama LLM
   - Creates the Chainlit interface

3. **.env**
   - Contains environment variables like the Qdrant URL

### Architecture Flow

The architecture follows these steps:

1. **Document Processing**:
   - PDF document → Docling extraction → Text chunks

2. **Embedding and Storage**:
   - Text chunks → Hugging Face embeddings → Qdrant vector database

3. **Query Processing**:
   - User question → Embedding → Semantic search → Relevant chunks

4. **Response Generation**:
   - Question + Relevant chunks → DeepSeek LLM → Natural language response

This implementation creates a fully local RAG system that can answer questions about your documents without requiring external API calls or cloud services.

## Project Conclusion

This project successfully demonstrates the implementation of a fully local Retrieval-Augmented Generation (RAG) system using DeepSeek and Ollama. By leveraging modern AI techniques and tools, we've created a solution that allows users to interact with PDF documents through natural language queries without relying on external APIs or cloud services.

### Challenges Encountered

Throughout the development of this local RAG implementation, several challenges were encountered and overcome:

1. **Local LLM Performance Constraints**: Running large language models locally comes with computational limitations. The DeepSeek model, while powerful, requires significant system resources. Finding the right balance between model size and performance was challenging, especially when running on consumer-grade hardware.

2. **Document Chunking Optimization**: Determining the optimal chunk size for document processing required experimentation. Chunks that are too small might lose context, while chunks that are too large might dilute the relevance of search results. The HybridChunker from Docling helped address this challenge by intelligently breaking down documents based on content structure.

3. **Vector Database Configuration**: Setting up and optimizing Qdrant for local use required careful configuration. Ensuring proper persistence of vector data between sessions and optimizing search parameters for the best retrieval performance took several iterations.

4. **Embedding Model Selection**: Choosing the right embedding model was crucial for effective semantic search. The "sentence-transformers/all-MiniLM-L6-v2" model was selected for its balance of performance and resource requirements, but this decision came after testing multiple alternatives.

5. **Integration Complexity**: Orchestrating the various components (document processing, embedding generation, vector storage, LLM integration, and UI) required careful planning and implementation. Ensuring smooth data flow between these components while maintaining a modular architecture was a significant challenge.

6. **Dependency Management**: Managing the dependencies for all the different components (LangChain, Chainlit, Docling, Hugging Face, Qdrant, Ollama) required careful version control to avoid compatibility issues.

### Lessons Learned

The development of this project provided several valuable insights and lessons:

1. **Local AI Power**: Modern consumer hardware is capable of running sophisticated AI applications locally. This opens up possibilities for privacy-preserving AI applications that don't rely on external services.

2. **Modular Architecture Benefits**: The modular approach used in this project (separating document processing, embedding, storage, and response generation) made the system more maintainable and adaptable. Each component could be optimized or replaced independently.

3. **Vector Databases for RAG**: Vector databases like Qdrant are essential for efficient semantic search in RAG applications. Their ability to quickly find similar vectors enables the retrieval of relevant context for user queries.

4. **Importance of Quality Embeddings**: The quality of embeddings significantly impacts the performance of the RAG system. Investing time in selecting and optimizing the embedding model pays dividends in retrieval accuracy.

5. **UI Considerations for AI Applications**: The Chainlit interface provides transparency into the RAG process, showing users how their questions are being processed. This transparency builds trust and helps users understand the system's capabilities and limitations.

6. **Docker for Dependency Isolation**: Using Docker for components like Qdrant simplified deployment and avoided potential conflicts with other system components.

7. **Prompt Engineering Matters**: Even with a well-designed RAG system, the way prompts are constructed for the LLM significantly impacts response quality. Crafting effective prompts that combine user questions with retrieved context is both an art and a science.

### Future Improvements

While the current implementation successfully demonstrates a local RAG system, several improvements could enhance its functionality and performance:

1. **Multi-Document Support**: Extend the system to handle multiple documents simultaneously, allowing users to query across a document collection rather than a single PDF.

2. **Improved Document Processing**: Incorporate better handling of tables, images, and other non-text elements in PDF documents to provide more comprehensive information retrieval.

3. **Hybrid Search Implementation**: Combine semantic search with keyword-based search for improved retrieval performance, especially for queries containing specific terms or numbers.

4. **User Feedback Integration**: Implement a feedback mechanism that allows users to rate responses and uses this feedback to improve future retrievals.

5. **Caching Mechanism**: Add response caching for frequently asked questions to improve performance and reduce computational load.

6. **Model Switching Capability**: Allow users to switch between different local LLMs based on their specific needs and available computational resources.

7. **Advanced Document Preprocessing**: Implement more sophisticated document preprocessing techniques, such as entity recognition and relationship extraction, to enhance the quality of information retrieval.

8. **Performance Optimization**: Optimize the system for better performance on lower-end hardware, making it accessible to more users.

9. **Conversation History Awareness**: Enhance the system to maintain conversation context across multiple queries, allowing for more natural follow-up questions.

10. **Customizable Chunking Strategies**: Provide options for users to select different chunking strategies based on their specific documents and use cases.

This project serves as a solid foundation for exploring and implementing local RAG systems. The modular architecture allows for continuous improvement and adaptation to different use cases and requirements. By leveraging the power of local LLMs and vector databases, we've demonstrated that sophisticated AI-powered document interaction is possible without relying on external services, providing both privacy and flexibility for users.

## References

- [Docling RAG with LangChain Example](https://ds4sd.github.io/docling/examples/rag_langchain/)
- [How to Build a Chatbot to Chat with Your PDF](https://blog.gopenai.com/how-to-build-a-chatbot-to-chat-with-your-pdf-9abb9beaf0c4)
- [Ollama Documentation](https://ollama.com/docs)
- [Qdrant Vector Database Documentation](https://qdrant.tech/documentation/)

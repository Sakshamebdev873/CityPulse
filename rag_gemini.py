import os
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aioconsole

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY environment variable not set.")
    raise SystemExit("GOOGLE_API_KEY environment variable not set.")

# Disable Chroma telemetry if specified
if os.getenv("CHROMA_TELEMETRY_ENABLED", "true").lower() == "false":
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    logger.info("Chroma telemetry disabled.")

# Configurable parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4))
EXCERPT_LENGTH = int(os.getenv("EXCERPT_LENGTH", 500))
EXPECTED_DATASET_SIZE = int(os.getenv("EXPECTED_DATASET_SIZE", 400))

async def main():
    # 1. Load dataset
    try:
        dataset = load_dataset("RealTimeData/bbc_news_alltime", "2025-06", split=f"train[:{EXPECTED_DATASET_SIZE}]")
        logger.info(f"Loaded {len(dataset)} items from dataset")
        if len(dataset) < EXPECTED_DATASET_SIZE:
            logger.warning(f"Dataset contains only {len(dataset)} items, expected {EXPECTED_DATASET_SIZE}. "
                          "Consider using a different dataset like 'ag_news' or adjusting EXPECTED_DATASET_SIZE.")
    except Exception as e:
        logger.error(f"Error loading dataset RealTimeData/bbc_news_alltime: {e}")
        logger.info("Falling back to 'ag_news' dataset as an alternative.")
        try:
            dataset = load_dataset("ag_news", split=f"train[:{EXPECTED_DATASET_SIZE}]")
            logger.info(f"Loaded {len(dataset)} items from ag_news dataset")
        except Exception as e:
            logger.error(f"Error loading fallback dataset: {e}")
            raise SystemExit("Failed to load any dataset. Ensure a valid dataset is available.")

    # 2. Prepare documents with metadata
    documents = []
    for item in dataset:
        if not item.get('content') and not item.get('text'):  # Handle both 'content' (BBC) and 'text' (ag_news)
            logger.warning(f"Skipping item with missing content: {item.get('title', 'No title')}")
            continue
        metadata = {
            'title': item.get('title', 'No title'),
            'url': item.get('url', 'No URL'),
            'source': item.get('source', 'BBC News' if 'content' in item else 'AG News')
        }
        for date_field in ['published', 'date', 'publish_date']:
            if date_field in item:
                metadata['date'] = item[date_field]
                break
        documents.append({
            'text': item.get('content') or item.get('text'),  # Handle both 'content' and 'text'
            'metadata': metadata
        })

    # 3. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = []
    for doc in documents:
        chunks = splitter.split_text(doc['text'])
        for chunk in chunks:
            splits.append({
                'page_content': chunk,
                'metadata': doc['metadata']
            })

    # 4. Initialize embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise SystemExit("Ensure GOOGLE_API_KEY is set.")

    # 5. Create or load vectorstore
    try:
        if os.path.exists("./bbc_news_chroma"):
            logger.info("Loading existing vector store from ./bbc_news_chroma")
            vectorstore = Chroma(persist_directory="./bbc_news_chroma", embedding_function=embeddings)
        else:
            logger.info("Creating new vector store")
            docs = [Document(page_content=split['page_content'], metadata=split['metadata']) 
                    for split in tqdm(splits, desc="Preparing documents")]
            vectorstore = await Chroma.afrom_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="./bbc_news_chroma"
            )
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise SystemExit("Failed to create vectorstore.")

    # 6. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 7. Define the prompt template
    template = """You are a news assistant. Answer the question based only on the following context.
    Include relevant excerpts and cite sources using the provided metadata.

    Context: {context}

    Question: {question}

    Provide a detailed answer with source attribution:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([
            f"Source {i+1} (Title: {doc.metadata.get('title', 'Unknown')}, " +
            f"Date: {doc.metadata.get('date', 'Unknown')})\n" +
            f"Content: {doc.page_content[:EXCERPT_LENGTH]}..."
            for i, doc in enumerate(docs)
        ])

    # 8. Initialize LLM
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro","gemini-2.0-flash"]
        if model_name not in available_models:
            logger.warning(f"Model {model_name} not recognized. Falling back to gemini-1.5-flash.")
            model_name = "gemini-1.5-flash"
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.4
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise SystemExit("Failed to initialize Gemini LLM.")

    # 9. Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 10. Interactive Q&A loop
    while True:
        try:
            query = (await aioconsole.ainput("\nAsk about news (or 'exit'): ")).strip()
            if query.lower() in ["exit", "quit"]:
                break

            logger.info(f"Processing query: {query}")
            print("\n🔍 Retrieving relevant documents...")
            retrieved_docs = await retriever.ainvoke(query)
            
            if not retrieved_docs:
                print("No relevant documents found. Try a different query.")
                continue
            
            print("\n📄 Retrieved documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Untitled article')}")
                if 'date' in doc.metadata:
                    print(f"   Date: {doc.metadata['date']}")
                print(f"   URL: {doc.metadata.get('url', 'No URL available')}")
                print(f"   Excerpt: {doc.page_content[:300]}...")
            
            print("\n🧠 Generating answer...")
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
            async def invoke_chain(query):
                return await rag_chain.ainvoke(query)
            
            answer = await invoke_chain(query)
            
            print("\n📰 Final Answer:")
            print(answer)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"⚠️ Error: {e}. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
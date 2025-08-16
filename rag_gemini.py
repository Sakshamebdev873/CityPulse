import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import asyncio

# Load environment variables
load_dotenv()

async def main():
    # 1. Load dataset
    try:
        dataset = load_dataset("RealTimeData/bbc_news_alltime","2025-06", split="train[:400]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise SystemExit("Failed to load dataset.")

    # 2. Prepare documents with metadata
    documents = []
    for item in dataset:
        if item['content']:
            metadata = {
                'title': item.get('title', 'No title'),
                'url': item.get('url', 'No URL'),
                'source': item.get('source', 'BBC News')
            }
            for date_field in ['published', 'date', 'publish_date']:
                if date_field in item:
                    metadata['date'] = item[date_field]
                    break
            
            documents.append({
                'text': item['content'],
                'metadata': metadata
            })

    # 3. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
        print(f"Error initializing embeddings: {e}")
        raise SystemExit("Ensure GOOGLE_API_KEY is set.")

    # 5. Create vectorstore
    try:
        from langchain_core.documents import Document
        docs = [Document(page_content=split['page_content'], metadata=split['metadata']) 
               for split in splits]
        
        vectorstore = await Chroma.afrom_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./bbc_news_chroma"
        )
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise SystemExit("Failed to create vectorstore.")

    # 6. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 7. Define the two-step process
    template = """You are a BBC News assistant. Answer the question based only on the following context.
    Include relevant excerpts and cite sources using the provided metadata.

    Context: {context}

    Question: {question}

    Provide a detailed answer with source attribution:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([
            f"Source {i+1} (Title: {doc.metadata.get('title', 'Unknown')}, " +
            f"Date: {doc.metadata.get('date', 'Unknown')})\n" +
            f"Content: {doc.page_content[:500]}..."
            for i, doc in enumerate(docs)
        ])

    # 8. Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise SystemExit("Failed to initialize Gemini LLM.")

    # 9. Create the two-step chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 10. Interactive Q&A loop
    try:
        while True:
            query = input("\nAsk about BBC News (or 'exit'): ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            try:
                print("\n🔍 Retrieving relevant documents...")
                retrieved_docs = await retriever.ainvoke(query)
                
                print("\n📄 Retrieved documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"\n{i}. {doc.metadata.get('title', 'Untitled article')}")
                    if 'date' in doc.metadata:
                        print(f"   Date: {doc.metadata['date']}")
                    print(f"   URL: {doc.metadata.get('url', 'No URL available')}")
                    print(f"   Excerpt: {doc.page_content[:300]}...")
                
                print("\n🧠 Generating answer...")
                answer = await rag_chain.ainvoke(query)
                
                print("\n📰 Final Answer:")
                print(answer)
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                
    finally:
        # Cleanup
        await vectorstore.adelete()
        await embeddings.adelete()
        if hasattr(llm, 'aclose'):
            await llm.aclose()

if __name__ == "__main__":
    asyncio.run(main())
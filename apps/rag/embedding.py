import os
import uuid

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from apps.rag.schemas import Vector, Chunk

load_dotenv()

embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" ,
    model=os.getenv("EMBEDDING_MODEL"),
    dimensions=int(os.getenv("DIMENSIONS")),
    check_embedding_ctx_length=False,
    chunk_size=10,

)


async def embedding_documents(chunks: list[Chunk]) -> list[Vector]:
    embedding_text = [chunk.content for chunk in chunks]
    vectors_plain = await embedding_model.aembed_documents(embedding_text)
    vectors = []
    for chunk, vector in zip(chunks, vectors_plain):
        vector_id = str(uuid.uuid4())
        chunk.vector_id = vector_id
        vectors.append(Vector(user_id=chunk.user_id,
                              vector_id=vector_id,
                              vector=vector,
                              document_id=chunk.document_id,
                              page_number=chunk.page_number,
                              chunk_index=chunk.chunk_index,
                              filename=chunk.filename
                              ))
    return vectors



async def embed_query(question: str) -> list[float]:
    return await embedding_model.aembed_query(question)
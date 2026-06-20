from langchain_text_splitters import RecursiveCharacterTextSplitter

from apps.rag.schemas import ParsedPage, Chunk

splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                          chunk_overlap=120,
                                          separators=["\n\n","\n","。","！","？", "."," ","",])



async def split_pages(pages: list[ParsedPage]) -> list[Chunk]:
    chunks = []
    chunk_index = 0
    for page in pages:
        contents = splitter.split_text(page.content)
        for index,chunk in enumerate(contents):
            chunk_index += 1
            chunks.append(Chunk(user_id=page.user_id,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                content=chunk,
                                document_id=page.document_id,
                                filename=page.filename))

    return  chunks
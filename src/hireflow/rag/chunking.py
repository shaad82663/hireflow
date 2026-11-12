from langchain_text_splitters import RecursiveCharacterTextSplitter


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=120,  
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in _splitter.split_text(text) if c.strip()]
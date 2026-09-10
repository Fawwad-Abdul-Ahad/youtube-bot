from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# def yt_Loader(video):
def video_process(video_id : str):
    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=kCc8FmEb1nY",
        language = ['en', 'ur', 'hi']
    )

    # Load documents
    documents = loader.load()
    # Transcript text
    # print(documents[0].page_content)  
    
    # splitting 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    # 4. Verify karein
    print(f"\n✅ Total chunks: {len(chunks)}")
    print(f"📊 First chunk length: {len(chunks[0].page_content)} chars")
    print(f"📄 First chunk preview: {chunks[0].page_content[:200]}...")
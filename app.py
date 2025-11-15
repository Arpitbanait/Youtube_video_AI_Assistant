import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from youtube_transcript_api import YouTubeTranscriptApi as YTTranscriptApi
import textwrap
import yt_dlp
import requests
import time
import re
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None

# Page configuration
st.set_page_config(
    page_title="YouTube Video Chat Assistant",
    page_icon="🎥",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                          help="Higher values make the output more creative but less focused")
    


    st.markdown("### Example URLs")
    st.markdown("""
    Try these example videos with subtitles:
    - TED Talk: `https://www.youtube.com/watch?v=eIho2S0ZahI`
    - Tutorial: `https://www.youtube.com/watch?v=VM6GUOFuey4`
    """)
    
    # Add help section
    with st.expander("ℹ️ Help"):
        st.markdown("""
        **Supported URL formats:**
        - Regular: `https://www.youtube.com/watch?v=VIDEO_ID`
        - Shortened: `https://youtu.be/VIDEO_ID`
        - Mobile: `https://m.youtube.com/watch?v=VIDEO_ID`
        
        **Common issues:**
        1. Make sure the video has subtitles/captions enabled
        2. Video should not be private or age-restricted
        3. Check if the URL is correctly copied
        
        **Tips:**
        - Use videos with clear speech and good captions
        - English language videos work best
        - Try educational content, lectures, or tutorials
        """)


# Main title with icon
st.title("🎬 YouTube Video Chat Assistant")
st.markdown("---")


def get_video_info(url, video_id, status):
    """Get video information with retries and fallbacks using yt-dlp"""
    try:
        # Configure yt-dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'format': 'best',
        }
        
        # Add retries for fetching video info
        retry_count = 3
        for attempt in range(retry_count):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_info = {
                        "title": info.get('title', 'Unknown Title'),
                        "author": info.get('uploader', 'Unknown Author'),
                        "description": info.get('description', ''),
                        "views": info.get('view_count', 0),
                        "thumbnail_url": info.get('thumbnail', f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"),
                        "video_id": video_id
                    }
                break
            except Exception as e:
                if attempt < retry_count - 1:
                    status.write(f"Retrying... (Attempt {attempt + 2}/{retry_count})")
                    time.sleep(1)  # Wait before retrying
                    continue
                raise e
        
        # Fallback for thumbnail if maxresdefault is not available
        try:
            response = requests.head(video_info["thumbnail_url"], timeout=5)
            if response.status_code != 200:
                video_info["thumbnail_url"] = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        except Exception:
            video_info["thumbnail_url"] = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        
        return video_info
    except Exception as e:
        error_msg = str(e)
        if "Video unavailable" in error_msg:
            st.error("This video is private or age-restricted.")
        elif "Video is age restricted" in error_msg:
            st.error("This video is age-restricted and cannot be accessed.")
        else:
            st.error(f"""Error fetching video information. This could be because:
            - The video is private or age-restricted
            - YouTube is rate-limiting requests
            - Network connectivity issues
            
            Try again in a few minutes or try a different video.
            Specific error: {error_msg}""")
        return None


def extract_video_id(url):
 
    patterns = [
        r'(?:v=|\\/)([0-9A-Za-z_-]{11}).*',  
        r'(?:embed\\/)([0-9A-Za-z_-]{11})',  
        r'(?:youtu\\.be\\/)([0-9A-Za-z_-]{11})' 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def clean_url(url):
    if not url:
        return ""
        
    # Remove any whitespace
    url = url.strip()
    
    # If it's a mobile URL, convert to desktop
    if 'm.youtube.com' in url:
        url = url.replace('m.youtube.com', 'www.youtube.com')
    
    # If it's a shortened URL, keep as is
    if 'youtu.be' in url:
        return url
    
    # For regular URLs, ensure they're in the correct format
    if 'youtube.com/watch?' in url:
        # Extract video ID and reconstruct URL
        video_id = extract_video_id(url)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return url


def process_video(url):
    try:
        # Extract video ID
        with st.status("Processing video...") as status:
            status.write("Extracting video ID...")
            video_id = extract_video_id(url)
            if not video_id:
                st.error("Could not extract video ID. Please check if the URL is correct.")
                return None
            
            status.write("✅ Successfully extracted video ID")
            
            # Get video info using yt-dlp
            status.write("Fetching video information...")
            video_info = get_video_info(url, video_id, status)
            if not video_info:
                return None
            status.write("✅ Successfully fetched video information")
            
            # Get transcript
            status.write("Fetching video transcript...")
            try:
                api = YTTranscriptApi()
                fetched_transcript = api.fetch(video_id)
                transcript_list = fetched_transcript.to_raw_data()
                transcript = " ".join([entry["text"] for entry in transcript_list])
                status.write("✅ Successfully fetched video transcript")
            except Exception as e:
                error_msg = str(e)
                if "No transcript" in error_msg:
                    st.error("This video has no captions/subtitles available.")
                elif "Translation" in error_msg:
                    st.error("Could not get transcript in the requested language.")
                else:
                    st.error(f"""
                    Error fetching transcript. This could be because:
                    - The video doesn't have captions/subtitles
                    - The captions are disabled
                    - The video is private or age-restricted
                    
                    Specific error: {error_msg}
                    """)
                return None

            # Process transcript
            status.write("Processing transcript...")
            try:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                status.write("✅ Successfully processed transcript")
                status.update(label="✅ Video processed successfully!", state="complete")
            except Exception as e:
                st.error(f"Error processing transcript: {str(e)}")
                return None

            return video_info
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def get_llm_response(question, context):
    token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    client = InferenceClient(
        model="google/gemma-2-2b-it",
        token=token
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Answer the question using ONLY the transcript context below.
        If answer isn't in context, say \"I don't know.\"\n
        Context:
        {context}

        Question: {question}
        """
    )
    
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=300,
        temperature=temperature
    )
    return response.choices[0].message["content"]


# YouTube URL input section
st.markdown("""
Enter a YouTube URL below. The video must:
- Be publicly accessible
- Have captions/subtitles enabled
- Not be age-restricted
""")

url_col1, url_col2 = st.columns([3, 1])
with url_col1:
    url_input = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
with url_col2:
    if st.button("🔄 Clear", help="Clear the current video and chat history"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.experimental_rerun()

# Process URL when provided
if url_input:
    try:
        # Clean and validate URL
        cleaned_url = clean_url(url_input)
        if not cleaned_url:
            st.error("Please enter a YouTube URL")
        elif cleaned_url != url_input:
            st.info(f"URL cleaned to: {cleaned_url}")
        
        # Check if URL seems valid
        if not any(x in cleaned_url for x in ['youtube.com/watch?v=', 'youtu.be/']):
            st.error("""
            Invalid URL format. Please use one of these formats:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            """)
        else:
            # Only process if URL has changed
            current_video_id = extract_video_id(cleaned_url)
            if (not st.session_state.video_info or 
                current_video_id != st.session_state.video_info.get('video_id')):
                st.session_state.video_info = process_video(cleaned_url)
            video_info = st.session_state.video_info
            
            if video_info:
                st.success("Video processed successfully!")
                
                # Display video information
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Video Information")
                    st.write(f"**Title:** {video_info.get('title', 'N/A')}")
                    st.write(f"**Author:** {video_info.get('author', 'N/A')}")
                    st.write(f"**Views:** {video_info.get('views', 'N/A'):,}")
                    if video_info.get('description'):
                        with st.expander("Show Description"):
                            st.write(video_info['description'])
                with col2:
                    if video_info.get('thumbnail_url'):
                        st.image(video_info['thumbnail_url'], use_column_width=True)
                
                # Chat interface
                st.subheader("💬 Chat with the Video")
                question = st.text_input("Ask a question about the video:", placeholder="What is the main topic?")
                
                if question:
                    with st.spinner("Thinking..."):
                        # Get relevant context
                        docs = st.session_state.vector_store.similarity_search(question, k=4)
                        context = "\n\n".join(doc.page_content for doc in docs)
                        
                        # Get response
                        response = get_llm_response(question, context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"question": question, "answer": response})
                
                # Display chat history
                if st.session_state.chat_history:
                    st.subheader("Chat History")
                    for chat in st.session_state.chat_history:
                        st.write(f"**Q:** {chat['question']}")
                        st.write(f"**A:** {chat['answer']}")
                        st.markdown("---")
            
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        st.session_state.video_info = None
        video_info = None

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and LangChain")
import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
from langchain.schema import Document
import base64
from datetime import datetime


class Summarizer:
    def __init__(self, api_key):
        self.llm = ChatGroq(model="llama3-8b-8192",
                            groq_api_key=api_key,
                            temperature=0.3)

    def get_prompt(self, language):
        return PromptTemplate(
            template=f"""
            Provide a detailed summary of the following content in 300 words.
            The summary should be in {language}.
            Content: {{text}}
            """,
            input_variables=["text"]
        )

    def summarize(self, docs, language="English"):
        prompt = self.get_prompt(language)
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain.run(docs)


def load_youtube_content(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'force_generic_extractor': True
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Video")
            description = info.get("description", "No description available.")
            transcript = info.get("subtitles", {}).get("en", [{}])[0].get("text", "") if info.get("subtitles") else ""
            return f"{title}\n\n{description}\n\nTranscript:\n{transcript}"
    except Exception as e:
        st.error(f"Failed to load YouTube content: {str(e)}")
        return "Could not load video content"


def load_website_content(url):
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load website content: {str(e)}")
        return []


def create_download_link(content, filename="summary.txt"):
    """Generates a download link for the summary"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Summary</a>'
    return href


def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    summarizer = Summarizer(groq_api_key)

    st.set_page_config(page_title="LangChain Enhanced Summarizer", page_icon="🌟")
    st.title("YouTube or Website Summarizer")
    st.write("Welcome! Summarize content from YouTube videos or websites in a more detailed manner.")

    st.sidebar.title("About This App")
    st.sidebar.info(
        "This app uses LangChain and the Llama 3 model from Groq API to provide detailed summaries. "
        "Simply enter a URL (YouTube or website) and get a concise summary!"
    )

    st.header("How to Use:")
    st.write("1. Enter the URL of a YouTube video or website you wish to summarize.")
    st.write("2. Select your preferred language for the summary.")
    st.write("3. Click **Summarize** to get a detailed summary.")
    st.write("4. Download your summary using the download button.")

    st.subheader("Enter the URL:")
    generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="https://example.com")

    # Language selection dropdown
    languages = [
        "English", "Spanish", "French", "German", "Italian",
        "Portuguese", "Russian", "Chinese", "Japanese", "Arabic",
        "Hindi", "Korean", "Dutch", "Turkish", "Polish"
    ]
    selected_language = st.selectbox("Select summary language", languages)

    if st.button("Summarize"):
        if not generic_url.strip():
            st.error("Please provide a URL to proceed.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL (YouTube or website).")
        else:
            try:
                with st.spinner(f"Generating summary in {selected_language}..."):
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        text_content = load_youtube_content(generic_url)
                        docs = [Document(page_content=text_content)]
                    else:
                        docs = load_website_content(generic_url)

                    if not docs or not docs[0].page_content.strip():
                        st.error("No content could be extracted from the URL.")
                        return

                    output_summary = summarizer.summarize(docs, selected_language)

                    # Store the summary in session state
                    st.session_state.summary = output_summary
                    st.session_state.summary_language = selected_language
                    st.session_state.summary_url = generic_url

                    st.subheader(f"Detailed Summary in {selected_language}:")
                    st.success(output_summary)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

    # Download button (only shows if there's a summary)
    if 'summary' in st.session_state and st.session_state.summary:
        st.markdown("---")
        st.subheader("Download Summary")

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.txt"

        # Create download link
        download_link = create_download_link(
            content=st.session_state.summary,
            filename=filename
        )

        st.markdown(download_link, unsafe_allow_html=True)

    st.sidebar.write("Developed with ❤️ by Shubham Sinha")


if __name__ == "__main__":
    main()
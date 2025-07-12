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

class Summarizer:
    def __init__(self, api_key):
        self.llm = ChatGroq(model="llama-3.2-11b-text-preview", groq_api_key=api_key)
        self.prompt = PromptTemplate(
            template="""
            Provide a detailed summary of the following content in 300 words:
            Content: {text}
            """,
            input_variables=["text"]
        )

    def summarize(self, docs):
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=self.prompt)
        return chain.run(docs)

def load_youtube_content(url):
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Video")
        description = info.get("description", "No description available.")
        return f"{title}\n\n{description}"

def load_website_content(url):
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    return loader.load()

def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    summarizer = Summarizer(groq_api_key)

    st.set_page_config(page_title="LangChain Enhanced Summarizer", page_icon="🌟")
    st.title("YouTube or Website Summarizer")
    st.write("Welcome! Summarize content from YouTube videos or websites in a more detailed manner.")

    st.sidebar.title("About This App")
    st.sidebar.info(
        "This app uses LangChain and the Llama 3.2 model from Groq API to provide detailed summaries. "
        "Simply enter a URL (YouTube or website) and get a concise summary!"
    )

    st.header("How to Use:")
    st.write("1. Enter the URL of a YouTube video or website you wish to summarize.")
    st.write("2. Click **Summarize** to get a detailed summary.")
    st.write("3. Enjoy the results!")

    st.subheader("Enter the URL:")
    generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="https://example.com")

    if st.button("Summarize"):
        if not generic_url.strip():
            st.error("Please provide a URL to proceed.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL (YouTube or website).")
        else:
            try:
                with st.spinner("Processing..."):
                    if "youtube.com" in generic_url:
                        text_content = load_youtube_content(generic_url)
                        docs = [Document(page_content=text_content)]
                    else:
                        docs = load_website_content(generic_url)
                    output_summary = summarizer.summarize(docs)
                    st.subheader("Detailed Summary:")
                    st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception occurred: {e}")

    st.sidebar.header("Features Coming Soon")
    st.sidebar.write("- Option to download summaries")
    st.sidebar.write("- Language selection for summaries")
    st.sidebar.write("- Summary length customization")
    st.sidebar.write("- Integration with other content platforms")
    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️ by Shubham Sinha")

if __name__ == "__main__":
    main()
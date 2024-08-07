import streamlit as st
from dotenv import load_dotenv
from helper_function import handle_userinput, get_conversation_chain, get_pdf_text, get_text_chunks, get_vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF's 📑")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'",type="pdf", accept_multiple_files=True)
        
        #process files if it's less then 10MB
        if st.button("Process"):
            if pdf_docs:
                valid_files = [pdf for pdf in pdf_docs if pdf.size <= 10 * 1024 * 1024] 
                if len(valid_files) != len(pdf_docs):
                    st.warning("One or more files exceed the 10 MB limit. Only files under 10 MB will be processed.")
                
                if valid_files:
                    with st.spinner("Processing"):
                        # get pdf text
                        raw_text = get_pdf_text(valid_files)
                        
                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        st.success("Files processed successfully.", icon="✅")
            else:
                st.warning("Please upload PDFs to process.")

if __name__ == '__main__':
    main()

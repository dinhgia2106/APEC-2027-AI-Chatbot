# Overview of the APEC Chatbot

## Chatbot Features

- **Question Answering Capabilities**:

  - General information about **APEC 2025** ([source](https://apec2025.kr/?menuno=1))
  - Schedules, including **high-level meetings**, **business forums**, **gala dinners**, and other official events
  - Details on **venues**, including cities, conference centers, and hotels
  - Guidelines for **entry procedures**, **visas**, **medical requirements**, and **domestic transportation**
  - Information on **Vietnamese and Phu Quoc culture**, covering cuisine, tourism, and side events

- **Multilingual Support**: At minimum, the chatbot supports both **Vietnamese** and **English**

- **Quick Reply Options**: Predefined topic selections for faster user interaction

---

## Deployment Instructions

1. Install required libraries using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the data embedding script:

   ```bash
   python backend/Embed/embed_data.py
   ```

3. Build the FAISS vector database:

   ```bash
   python backend/VectorStore/build_faiss_index.py
   ```

4. Download and install **Ollama** from [this link](https://ollama.com/download)

5. Start the LLaMA 3 model:

   ```bash
   ollama run llama3
   ```

6. Test the retrieval-augmented generation (RAG) pipeline:

   ```bash
   python demo/test_rag_pipeline.py
   ```

7. Launch the Streamlit web interface:

   ```bash
   streamlit run demo/streamlit_app.py
   ```

---

## Notes

- Ensure all dependencies are properly installed before running the scripts.
- The chatbot system uses **FAISS** for efficient similarity search over embedded data.
- The application is designed for both **command-line testing** and **web-based interaction** via **Streamlit**.

### **Project To-Do List: APEC 2027 AI Chatbot Demo**

#### **Phase 0: Project Setup & Initialization**

- [x] Initialize a new Git repository on GitHub.
- [x] Set up the local Python virtual environment (e.g., using `venv` or `conda`).
- [x] Create the initial project directory structure:
  - `backend/`
  - `demo/`
- [ ] Create a `requirements.txt` file and add initial libraries (`fastapi`, `streamlit`, `langchain`, `openai`, etc.).
- [x] Create a `.gitignore` file to exclude virtual environments, `.env` files, and `__pycache__`.

---

#### **Phase 1: Data Collection & Preparation (Backend)**

- [ ] **Crawl Data:**
  - [ ] Write a script (`scripts/crawl.py`) using `BeautifulSoup` to crawl text from the APEC 2025 Korea website (`https://apec2025.kr/`).
  - [ ] Target specific sections: Overview, Program/Schedule, Venues.
  - [ ] Save the raw output as text files.
- [ ] **Create Mock Data:**
  - [ ] Write supplementary `.txt` files for topics not available online:
    - Mock Vietnamese visa/entry procedures.
    - Information on Phu Quoc (culture, food, key locations).
    - A mock high-level schedule for APEC 2027.
- [ ] **Process & Translate Data:**
  - [ ] Clean the crawled text to remove HTML tags and unwanted artifacts.
  - [ ] Translate the English content into Vietnamese to create a parallel dataset.
  - [ ] Organize all cleaned text files into language-specific folders: `data/en/` and `data/vi/`.
- [ ] **Build Vector Stores:**
  - [ ] Create a script (`scripts/prepare_vectorstore.py`).
  - [ ] This script must:
    - [ ] Load all text documents from `data/en` and `data/vi`.
    - [ ] Split documents into smaller, manageable chunks (using a text splitter).
    - [ ] Use a sentence-transformer model to create embeddings for all chunks.
    - [ ] Create and save two separate vector stores (e.g., using FAISS or ChromaDB): one for English (`vectorstore_en`) and one for Vietnamese (`vectorstore_vi`).

---

#### **Phase 2: AI Backend Development (FastAPI)**

- [ ] **Setup FastAPI Server:**
  - [ ] Inside the `backend/` directory, create the main application file (`main.py`).
  - [ ] Set up a `.env` file to store the OpenAI API key and other configurations.
  - [ ] Load environment variables securely into the application.
- [ ] **Implement RAG Logic:**
  - [ ] In a separate module (e.g., `backend/chatbot_service.py`), implement the core RAG functionality.
  - [ ] Write functions to load the LLM (e.g., `ChatOpenAI`) and the pre-built vector stores.
  - [ ] Create a prompt template that takes a `question` and `context` as input.
  - [ ] Build the RAG chain that connects the retriever, prompt, and LLM.
- [ ] **Create API Endpoint:**
  - [ ] Define a `/chat` POST endpoint in `main.py`.
  - [ ] Use Pydantic models to define the expected request body (e.g., `{ "message": str, "language": "en" | "vi" }`).
  - [ ] The endpoint logic should:
    1.  Select the correct vector store based on the `language` field.
    2.  Retrieve relevant context for the user's `message`.
    3.  Invoke the RAG chain to generate a response.
    4.  Return the response as JSON.
- [ ] **(Optional) Implement Language Detection:**
  - [ ] Add a language detection library (`langdetect`) to the `requirements.txt`.
  - [ ] Modify the `/chat` endpoint to automatically detect the language if it's not provided in the request.

---

#### **Phase 3: Frontend UI Development & Integration (Streamlit)**

- [ ] **Setup Streamlit App:**
  - [ ] Inside the `demo/` directory, create the application file (`app.py`).
  - [ ] Design a simple, clean title and layout for the chatbot.
- [ ] **Build Chat Interface:**
  - [ ] Use `st.session_state` to initialize and store the conversation history.
  - [ ] Loop through the session state to display past messages.
  - [ ] Use `st.text_input` to capture the user's new message.
- [ ] **Implement Quick Replies:**
  - [ ] Add three or more static buttons at the bottom or side of the app using `st.button` (e.g., "Today's schedule," "Tell me about Phu Quoc").
  - [ ] When a button is clicked, treat its label as the user's input.
- [ ] **Connect to Backend:**
  - [ ] When the user sends a message (either by typing or clicking a quick reply), use the `requests` library to make a POST call to the backend's `/chat` endpoint.
  - [ ] Append both the user's message and the bot's returned reply to the `st.session_state` history.
  - [ ] Re-run the Streamlit app to refresh the display with the new messages.

---

#### **Phase 4: Testing, Documentation, and Finalization**

- [ ] **Testing:**
  - [ ] Run the entire application locally (backend server + Streamlit app).
  - [ ] Test with a variety of questions in both English and Vietnamese to check for accuracy and relevance.
  - [ ] Test the quick reply functionality.
  - [ ] **(Optional) Map Link Test:** If implementing map integration, ask a location-based question and verify that a correct Google Maps URL is returned and is clickable.
- [ ] **Documentation:**
  - [ ] Create a high-quality `README.md` file in the root directory.
  - [ ] The `README.md` must include:
    - [ ] A clear project overview.
    - [ ] **Installation Guide:** How to set up the environment and install dependencies from `requirements.txt`.
    - [ ] **Configuration:** How to create the `.env` file and add the API key.
    - [ ] **How to Run:** A step-by-step guide (e.g., `1. Run the data preparation script`, `2. Start the FastAPI backend`, `3. Run the Streamlit demo`).
    - [ ] **Example Conversation:** A screenshot or text block showing a sample interaction with the chatbot.
- [ ] **Code & Repository Cleanup:**
  - [ ] Review all code for clarity and add comments where necessary.
  - [ ] Ensure the repository structure is clean and logical.
  - [ ] Make a final commit with the complete, working project.

---

### **Final Deliverable**

- [ ] A public GitHub repository containing the `backend/`, `demo/`, `scripts/`, `data/` folders and the final `README.md`.

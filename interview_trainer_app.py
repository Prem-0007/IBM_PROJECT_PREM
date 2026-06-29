import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()
API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

if API_KEY is None:
    try:
        API_KEY = st.secrets["IBM_API_KEY"]
        PROJECT_ID = st.secrets["PROJECT_ID"]
    except Exception:
        pass

INTERVIEW_DATA = """
Python Developer Interview Questions:
Q: What is a decorator in Python?
A: A decorator is a function that wraps another function, extending its behavior without modifying the original code. Common examples are @staticmethod, @classmethod, and @property.

Q: What is the difference between list and tuple?
A: Lists are mutable (can be changed), tuples are immutable (cannot be changed). Tuples are faster and used for fixed data.

Q: What is OOP in Python?
A: Object Oriented Programming uses classes, objects, inheritance, polymorphism, and encapsulation to model real-world entities.

Q: What is a lambda function?
A: A lambda is a small anonymous function defined with the lambda keyword. Example: square = lambda x: x**2

Q: What is the GIL in Python?
A: The Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously, limiting true parallelism in CPython.

Q: What is a generator in Python?
A: A generator is a function that yields values lazily using the yield keyword, saving memory compared to returning lists.

Q: What is the difference between deepcopy and shallow copy?
A: Shallow copy copies object references, deepcopy copies all nested objects recursively.

Q: What are Python's built-in data types?
A: int, float, str, bool, list, tuple, dict, set, frozenset, bytes, bytearray, NoneType.

Q: What is a context manager?
A: An object that defines __enter__ and __exit__ methods used with the with statement for resource management.

Q: What is list comprehension?
A: A concise way to create lists: [x**2 for x in range(10) if x % 2 == 0]

Data Science Interview Questions:
Q: What is overfitting?
A: Overfitting is when a model learns training data too well, including noise, and fails to generalize to new data.

Q: What is cross-validation?
A: A technique to evaluate ML models by splitting data into k folds, training on k-1 and testing on 1, repeated k times.

Q: What is the difference between supervised and unsupervised learning?
A: Supervised learning uses labeled data to learn mappings; unsupervised learning discovers hidden patterns in unlabeled data.

Q: What is a confusion matrix?
A: A 2x2 table showing TP, FP, TN, FN counts used to evaluate classification model performance.

Q: What is regularization?
A: Techniques like L1 (Lasso) and L2 (Ridge) regularization add penalties to prevent overfitting.

Q: What is the bias-variance tradeoff?
A: High bias = underfitting (model too simple), high variance = overfitting (model too complex). Goal is to balance both.

Q: What is PCA?
A: Principal Component Analysis reduces dimensionality by finding directions of maximum variance in data.

Q: What is gradient descent?
A: An optimization algorithm that iteratively moves in the direction of steepest descent of the loss function.

Q: What is feature engineering?
A: The process of creating new features or transforming existing ones to improve model performance.

Q: What is the difference between bagging and boosting?
A: Bagging trains models in parallel on random subsets (e.g. Random Forest); boosting trains sequentially to correct errors (e.g. XGBoost).

Web Development Interview Questions:
Q: What is REST API?
A: REST (Representational State Transfer) is an architectural style using HTTP methods: GET, POST, PUT, DELETE.

Q: What is the difference between GET and POST?
A: GET retrieves data and is idempotent; POST sends data to create/update resources and is not idempotent.

Q: What is JWT?
A: JSON Web Token is a compact, URL-safe token used for stateless authentication between client and server.

Q: What is CORS?
A: Cross-Origin Resource Sharing is a browser security mechanism that controls requests from different domains.

Q: What is the difference between SQL and NoSQL?
A: SQL is relational with a fixed schema (MySQL, PostgreSQL); NoSQL is non-relational and schema-flexible (MongoDB, Redis).

Q: What is Docker?
A: Docker is a containerization platform that packages applications with their dependencies for consistent deployment.

Q: What is CI/CD?
A: Continuous Integration and Continuous Deployment automates testing and deploying code changes.

Q: What is HTTP vs HTTPS?
A: HTTP is unencrypted; HTTPS uses TLS/SSL to encrypt data in transit for security.

Q: What is GraphQL?
A: A query language for APIs that lets clients request exactly the data they need, reducing over-fetching.

Q: What is a microservice?
A: An architectural pattern where an application is divided into small, independently deployable services.

Java Developer Interview Questions:
Q: What is polymorphism?
A: Polymorphism allows one interface to represent different types through method overriding (runtime) and overloading (compile-time).

Q: What is the difference between abstract class and interface?
A: Abstract classes can have implementations and state; interfaces (before Java 8) only define contracts.

Q: What is garbage collection in Java?
A: The JVM automatically deallocates memory for objects with no references using GC algorithms like G1, CMS.

Q: What is the difference between == and equals()?
A: == compares references; equals() compares object content/value.

Q: What is a HashMap in Java?
A: A data structure that stores key-value pairs with O(1) average lookup using hashing.

Q: What is the difference between ArrayList and LinkedList?
A: ArrayList is backed by an array (fast access, slow insert/delete); LinkedList is node-based (slow access, fast insert/delete).

Q: What are Java streams?
A: Streams provide functional-style operations on collections: filter, map, reduce, collect.

Q: What is multithreading in Java?
A: Running multiple threads concurrently to improve performance, managed via Thread class or ExecutorService.

DevOps Interview Questions:
Q: What is Kubernetes?
A: Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management.

Q: What is Infrastructure as Code?
A: Managing infrastructure through code (Terraform, Ansible) instead of manual configuration.

Q: What is a load balancer?
A: A device or service that distributes incoming traffic across multiple servers for high availability.

Q: What is the difference between blue-green and canary deployments?
A: Blue-green switches all traffic instantly to new version; canary gradually shifts traffic to reduce risk.

Behavioral Interview Questions:
Q: Tell me about yourself.
A: Start with your current role, mention 2-3 key skills with impact, and end with why you want this specific job.

Q: What is your greatest strength?
A: Choose a strength relevant to the job. Back it with a real achievement. Example: "Problem-solving - I reduced API response time by 40%."

Q: Where do you see yourself in 5 years?
A: Show ambition aligned with company growth. Mention skills you want to develop and how you can contribute.

Q: Why do you want to work here?
A: Research the company's products, culture, and projects. Mention 2 specific reasons why this role excites you.

Q: How do you handle pressure?
A: Use STAR method: Situation, Task, Action, Result. Give a real example that shows composure and effectiveness.

Q: Tell me about a time you failed.
A: Choose a real failure, own it, explain what you learned, and show how you applied that lesson.

Q: How do you handle conflicts with teammates?
A: Describe active listening, finding common ground, focusing on the issue not the person.

Q: What motivates you?
A: Be genuine: learning new technologies, delivering impact, solving complex problems, team collaboration.
"""


@st.cache_resource(show_spinner=False)
def load_model():
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        api_key=API_KEY
    )
    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials=credentials,
        project_id=PROJECT_ID,
        params={"max_new_tokens": 700, "temperature": 0.7}
    )
    return model


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.create_documents([INTERVIEW_DATA])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_response(model, vectorstore, job_role, question):
    docs = vectorstore.similarity_search(f"{job_role} {question}", k=4)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an expert Interview Trainer helping candidates prepare for {job_role} interviews.

Relevant Knowledge:
{context}

Question: {question}

Provide:
1. Direct Answer (2-3 sentences)
2. Interview Tip (1 actionable tip)
3. Sample Answer (ready-to-use, 3-4 sentences)

Keep it concise and practical."""
    return model.generate_text(prompt)


def generate_questions(model, job_role, experience, difficulty, num_questions):
    prompt = f"""You are an expert Interview Trainer.

Generate {num_questions} {difficulty}-level interview questions for a {job_role} with {experience} experience.

For each question:
- Question
- Skill tested
- Model answer (3-4 sentences)

Format clearly. Be specific to {job_role}."""
    return model.generate_text(prompt)


def evaluate_answer(model, question, user_answer, job_role):
    prompt = f"""You are an expert Interview Trainer evaluating a {job_role} candidate.

Question: {question}
Candidate Answer: {user_answer}

Evaluate and provide:
1. Score: X/10
2. Strengths (bullet points)
3. Improvements (bullet points)
4. Stronger Version (rewritten answer)

Be constructive and specific."""
    return model.generate_text(prompt)


def mock_interview(model, job_role, experience):
    prompt = f"""You are conducting a mock interview for a {job_role} with {experience} experience.

Ask 1 interview question (mix of technical and behavioral).
After the question, provide:
- Difficulty: Easy/Medium/Hard
- Topic: what it covers
- Hint: one subtle hint without giving the answer away

Format:
**Question:** [question]
**Difficulty:** [level]
**Topic:** [topic]
**Hint:** [hint]"""
    return model.generate_text(prompt)


def resume_tips(model, job_role, experience):
    prompt = f"""You are a resume expert helping a {job_role} candidate with {experience} experience.

Provide:
1. Top 5 resume tips specific to {job_role}
2. 3 strong action verbs to use
3. 2 bullet point examples they can adapt
4. Common mistakes to avoid

Be specific and actionable."""
    return model.generate_text(prompt)


def salary_negotiation(model, job_role, experience, location):
    prompt = f"""You are a career coach helping a {job_role} with {experience} experience in {location}.

Provide:
1. Salary range estimate for this profile
2. How to negotiate confidently (3 steps)
3. What to say when asked "What's your expected salary?"
4. Phrases to use and avoid during negotiation

Be practical and direct."""
    return model.generate_text(prompt)


def add_to_history(role, question, answer, score=None):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M"),
        "role": role,
        "question": question,
        "answer": answer,
        "score": score
    })


# ─── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: white;
    }
    .history-item {
        border-left: 3px solid #0f3460;
        padding: 8px 12px;
        margin: 8px 0;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Interview Trainer Agent")
st.markdown("**PS No. 22 | IBM SkillsBuild AICTE 2026 | Powered by IBM watsonx.ai + RAG**")
st.markdown("---")

if "history" not in st.session_state:
    st.session_state.history = []
if "mock_question" not in st.session_state:
    st.session_state.mock_question = ""

with st.spinner("Loading AI models and embeddings..."):
    model = load_model()
    vectorstore = load_vectorstore()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=90)
st.sidebar.title("⚙️ Configuration")

job_role = st.sidebar.selectbox(
    "Job Role",
    ["Python Developer", "Data Scientist", "Web Developer", "Java Developer", "DevOps Engineer", "General HR"]
)
experience = st.sidebar.selectbox(
    "Experience",
    ["Fresher", "1 year", "2-3 years", "5+ years"]
)
difficulty = st.sidebar.select_slider(
    "Difficulty",
    options=["Easy", "Medium", "Hard"],
    value="Medium"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Stats")
col1, col2 = st.sidebar.columns(2)
col1.metric("Questions", len(st.session_state.history))
scores = [h["score"] for h in st.session_state.history if h["score"]]
col2.metric("Avg Score", f"{sum(scores)/len(scores):.1f}" if scores else "—")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Stack")
st.sidebar.markdown("""
- IBM watsonx.ai
- Meta LLaMA 3.3 70B  
- RAG + FAISS
- HuggingFace Embeddings
- LangChain + Streamlit
""")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Q&A Assistant",
    "📋 Generate Questions",
    "✅ Evaluate Answer",
    "🎭 Mock Interview",
    "📄 Resume Tips",
    "💰 Salary Guide"
])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 💬 Ask Anything About Your Interview")
    st.caption("Uses real semantic search over the knowledge base to find relevant context.")

    user_input = st.text_input(
        "Your question",
        placeholder="How do I answer 'what is your weakness'?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("Ask 🚀", use_container_width=True)

    if ask_btn:
        if user_input:
            with st.spinner("Searching knowledge base and generating answer..."):
                t0 = time.time()
                response = get_response(model, vectorstore, job_role, user_input)
                elapsed = time.time() - t0
            st.success(f"Answer generated in {elapsed:.1f}s")
            st.markdown("### 💡 Answer")
            st.markdown(response)
            add_to_history(job_role, user_input, response)
        else:
            st.warning("Enter a question first.")

# ── Tab 2: Generate Questions ─────────────────────────────────────────────────
with tab2:
    st.markdown("### 📋 Generate Interview Questions")

    col1, col2 = st.columns(2)
    with col1:
        num_q = st.slider("Number of questions", 3, 10, 5)
    with col2:
        st.info(f"**Role:** {job_role} | **Level:** {experience} | **Difficulty:** {difficulty}")

    if st.button("Generate Questions 📝", use_container_width=True):
        with st.spinner("Generating personalized questions..."):
            response = generate_questions(model, job_role, experience, difficulty, num_q)
        st.markdown("### 📝 Your Question Set")
        st.markdown(response)
        add_to_history(job_role, f"Generate {num_q} {difficulty} questions", response)

        st.download_button(
            "⬇️ Download Questions",
            data=response,
            file_name=f"{job_role.replace(' ', '_')}_questions.txt",
            mime="text/plain"
        )

# ── Tab 3: Evaluate ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ✅ Evaluate Your Answer")

    question_input = st.text_input(
        "Interview Question",
        placeholder="What is a decorator in Python?"
    )
    answer_input = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        height=140
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        word_count = len(answer_input.split()) if answer_input else 0
        st.caption(f"Word count: {word_count}")

    if st.button("Evaluate 🎯", use_container_width=True):
        if question_input and answer_input:
            with st.spinner("Evaluating your answer..."):
                response = evaluate_answer(model, question_input, answer_input, job_role)
            st.markdown("### 📊 Evaluation")
            st.markdown(response)

            score = None
            for line in response.split("\n"):
                if "Score:" in line or "/10" in line:
                    import re
                    m = re.search(r"(\d+)/10", line)
                    if m:
                        score = int(m.group(1))
                        break

            if score:
                color = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                st.metric("Your Score", f"{color} {score}/10")

            add_to_history(job_role, question_input, answer_input, score)
        else:
            st.warning("Fill in both fields.")

# ── Tab 4: Mock Interview ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🎭 Mock Interview Session")
    st.caption("Get a fresh question, answer it, then get AI feedback.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Get New Question", use_container_width=True):
            with st.spinner("Picking a question..."):
                q = mock_interview(model, job_role, experience)
            st.session_state.mock_question = q

    if st.session_state.mock_question:
        st.markdown("---")
        st.markdown(st.session_state.mock_question)
        st.markdown("---")

        mock_answer = st.text_area("Your Answer", height=120, key="mock_ans")

        if st.button("Submit Answer 📤", use_container_width=True):
            if mock_answer:
                question_text = ""
                for line in st.session_state.mock_question.split("\n"):
                    if "**Question:**" in line:
                        question_text = line.replace("**Question:**", "").strip()
                        break

                with st.spinner("Evaluating..."):
                    feedback = evaluate_answer(model, question_text or "Mock Interview Question", mock_answer, job_role)
                st.markdown("### 🔍 Feedback")
                st.markdown(feedback)
            else:
                st.warning("Type your answer first.")

# ── Tab 5: Resume Tips ────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📄 Resume Tips for Your Role")

    st.info(f"Getting tailored tips for **{job_role}** with **{experience}** experience")

    if st.button("Get Resume Tips ✨", use_container_width=True):
        with st.spinner("Generating resume advice..."):
            response = resume_tips(model, job_role, experience)
        st.markdown("### 💼 Resume Guide")
        st.markdown(response)

        st.download_button(
            "⬇️ Download Tips",
            data=response,
            file_name=f"{job_role.replace(' ', '_')}_resume_tips.txt",
            mime="text/plain"
        )

# ── Tab 6: Salary Guide ───────────────────────────────────────────────────────
with tab6:
    st.markdown("### 💰 Salary & Negotiation Guide")

    location = st.selectbox(
        "Location",
        ["Bangalore, India", "Hyderabad, India", "Mumbai, India", "Delhi, India",
         "Chennai, India", "Pune, India", "Remote (India)", "USA", "UK", "Singapore"]
    )

    if st.button("Get Salary Guide 💰", use_container_width=True):
        with st.spinner("Researching salary data..."):
            response = salary_negotiation(model, job_role, experience, location)
        st.markdown("### 📈 Salary & Negotiation")
        st.markdown(response)

# ── History Panel ─────────────────────────────────────────────────────────────
if st.session_state.history:
    with st.expander(f"📜 Session History ({len(st.session_state.history)} items)", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history)):
            score_str = f" | Score: {item['score']}/10" if item["score"] else ""
            st.markdown(f"""
<div class="history-item">
<b>#{len(st.session_state.history)-i}</b> [{item['timestamp']}] {item['role']}{score_str}<br>
<small>Q: {item['question'][:80]}{'...' if len(item['question']) > 80 else ''}</small>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("*🎯 Interview Trainer Agent | IBM watsonx.ai | AICTE 2026 IBM SkillsBuild | PS No. 22*")

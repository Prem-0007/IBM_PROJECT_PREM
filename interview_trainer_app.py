import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Load environment variables
load_dotenv()
API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Fallback for Streamlit Cloud secrets
if API_KEY is None:
    try:
        API_KEY = st.secrets["IBM_API_KEY"]
        PROJECT_ID = st.secrets["PROJECT_ID"]
    except:
        pass

# Setup IBM watsonx.ai
@st.cache_resource
def load_model():
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        api_key=API_KEY
    )
    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials=credentials,
        project_id=PROJECT_ID,
        params={"max_new_tokens": 500, "temperature": 0.7}
    )
    return model

# Setup RAG
@st.cache_resource
def load_vectorstore():
    interview_data = """
Python Developer Interview Questions:
Q: What is a decorator in Python?
A: A decorator is a function that takes another function and extends its behavior.

Q: What is the difference between list and tuple?
A: Lists are mutable, tuples are immutable.

Q: What is OOP in Python?
A: Object Oriented Programming with classes, inheritance, polymorphism, encapsulation.

Q: What is a lambda function?
A: A lambda is an anonymous function defined with the lambda keyword.

Q: What is the GIL in Python?
A: Global Interpreter Lock prevents multiple threads from executing Python code simultaneously.

Q: What is a generator in Python?
A: A generator is a function that yields values one at a time using the yield keyword.

Data Science Interview Questions:
Q: What is overfitting?
A: When a model performs well on training data but poorly on new data.

Q: What is cross-validation?
A: A technique to evaluate ML models by training on subsets of data.

Q: What is the difference between supervised and unsupervised learning?
A: Supervised uses labeled data, unsupervised finds patterns in unlabeled data.

Q: What is a confusion matrix?
A: A table showing true positives, false positives, true negatives, false negatives.

Web Development Interview Questions:
Q: What is REST API?
A: REST is an architectural style for building web services using HTTP methods.

Q: What is the difference between GET and POST?
A: GET retrieves data, POST sends data to the server.

Q: What is JWT?
A: JSON Web Token used for secure authentication between client and server.

Java Developer Interview Questions:
Q: What is polymorphism?
A: Ability of an object to take multiple forms through method overriding and overloading.

Q: What is the difference between abstract class and interface?
A: Abstract class can have method implementations, interface cannot before Java 8.

Q: What is garbage collection in Java?
A: Automatic memory management that removes unused objects from heap memory.

Behavioral Interview Questions:
Q: Tell me about yourself.
A: Start with current role, mention key skills, end with why you want this job.

Q: What is your greatest strength?
A: Choose a strength relevant to the job and back it with a real example.

Q: Where do you see yourself in 5 years?
A: Show ambition aligned with the company growth opportunities.

Q: Why do you want to work here?
A: Research the company and mention specific reasons like culture, projects, or values.

Q: How do you handle pressure?
A: Give a real example using STAR method - Situation Task Action Result.
"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.create_documents([interview_data])
    embeddings = FakeEmbeddings(size=768)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_response(model, vectorstore, job_role, question):
    docs = vectorstore.similarity_search(f"{job_role} {question}", k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an expert Interview Trainer Agent helping candidates prepare for job interviews.

Job Role: {job_role}

Relevant Knowledge Base:
{context}

User Question: {question}

Provide a structured response with:
1. Direct Answer
2. Interview Tip
3. Example Answer the candidate can use

Keep it concise and practical."""
    return model.generate_text(prompt)

def generate_questions(model, job_role, experience):
    prompt = f"""You are an expert Interview Trainer Agent.

Generate 5 interview questions for a {job_role} with {experience} experience.

For each question provide:
- Question
- What skill it tests
- Model Answer

Format it clearly and practically."""
    return model.generate_text(prompt)

def evaluate_answer(model, question, user_answer, job_role):
    prompt = f"""You are an expert Interview Trainer Agent evaluating a candidate's answer.

Job Role: {job_role}
Interview Question: {question}
Candidate's Answer: {user_answer}

Evaluate the answer and provide:
1. Score out of 10
2. What was good
3. What to improve
4. Better version of the answer

Be constructive and encouraging."""
    return model.generate_text(prompt)

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide"
)

# Header
st.title("🎯 Interview Trainer Agent")
st.markdown("**PS No. 22 | IBM SkillsBuild AICTE 2026 | Powered by IBM watsonx.ai + RAG**")
st.markdown("---")

# Load resources
with st.spinner("Loading AI models..."):
    model = load_model()
    vectorstore = load_vectorstore()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
st.sidebar.title("⚙️ Settings")
job_role = st.sidebar.selectbox(
    "Select Job Role",
    ["Python Developer", "Data Scientist", "Web Developer", "Java Developer", "General HR"]
)
experience = st.sidebar.selectbox(
    "Experience Level",
    ["Fresher", "1 year", "2-3 years", "5+ years"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Tech Stack")
st.sidebar.markdown("- IBM watsonx.ai")
st.sidebar.markdown("- Meta LLaMA 3.3 70B")
st.sidebar.markdown("- RAG + FAISS")
st.sidebar.markdown("- LangChain")
st.sidebar.markdown("- Streamlit")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Ask Questions",
    "📋 Generate Questions",
    "✅ Evaluate My Answer"
])

# Tab 1 - Ask Questions
with tab1:
    st.markdown("### 💬 Ask Anything About Your Interview")
    user_input = st.text_input(
        "Your question:",
        placeholder="How do I answer questions about my weakness?"
    )
    if st.button("Get Answer 🚀", key="ask"):
        if user_input:
            with st.spinner("Thinking..."):
                response = get_response(model, vectorstore, job_role, user_input)
            st.markdown("### 💡 Answer:")
            st.markdown(response)
        else:
            st.warning("Please enter a question!")

# Tab 2 - Generate Questions
with tab2:
    st.markdown("### 📋 Generate Interview Questions")
    st.markdown(f"Generating for: **{job_role}** | Experience: **{experience}**")
    if st.button("Generate Questions 📝", key="generate"):
        with st.spinner("Generating questions..."):
            response = generate_questions(model, job_role, experience)
        st.markdown("### 📝 Interview Questions:")
        st.markdown(response)

# Tab 3 - Evaluate Answer
with tab3:
    st.markdown("### ✅ Evaluate Your Answer")
    question_input = st.text_input(
        "Interview Question:",
        placeholder="What is a decorator in Python?"
    )
    answer_input = st.text_area(
        "Your Answer:",
        placeholder="Type your answer here...",
        height=150
    )
    if st.button("Evaluate My Answer 🎯", key="evaluate"):
        if question_input and answer_input:
            with st.spinner("Evaluating your answer..."):
                response = evaluate_answer(model, question_input, answer_input, job_role)
            st.markdown("### 📊 Evaluation:")
            st.markdown(response)
        else:
            st.warning("Please enter both question and your answer!")

# Footer
st.markdown("---")
st.markdown("*🎯 Interview Trainer Agent | Built with IBM watsonx.ai | AICTE 2026 IBM SkillsBuild*")
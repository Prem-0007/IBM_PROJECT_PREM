# 🎯 Interview Trainer Agent
### IBM SkillsBuild AICTE 2026 | Problem Statement No. 22

An AI-powered Interview Trainer Agent built using IBM watsonx.ai and RAG (Retrieval-Augmented Generation) that helps students and job seekers prepare for interviews.

---

## 🚀 Live Demo
[Click here to open the app](https://ibmprojectprem-haha.streamlit.app/)

---

## 📌 Features
- 💬 Ask any interview-related question
- 📋 Generate role-specific interview questions
- ✅ Evaluate your answers and get feedback
- 🎯 Supports Python, Data Science, Web Dev, Java, HR roles
- 📊 RAG-based retrieval for accurate responses

---

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| IBM watsonx.ai | AI model hosting |
| Meta LLaMA 3.3 70B | Text generation |
| RAG + FAISS | Knowledge retrieval |
| LangChain | Agent pipeline |
| Streamlit | Frontend UI |
| Python | Backend logic |

---

## ⚙️ Setup Locally

1. Clone the repo:
```bash
git clone https://github.com/Prem-0007/IBM_PROJECT_PREM.git
cd IBM_PROJECT_PREM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
IBM_API_KEY=your_api_key_here

PROJECT_ID=your_project_id_here

4. Run the app:
```bash
python -m streamlit run interview_trainer_app.py
```

---

## 📁 Project Structure
IBM_PROJECT_PREM/

├── interview_trainer_app.py   # Main app

├── requirements.txt           # Dependencies

├── .gitignore                 # Git ignore

└── README.md                  # Documentation



*Built with ❤️ using IBM watsonx.ai | AICTE 2026 IBM SkillsBuild*

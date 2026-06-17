- 📰 Detect fake new# VERITAS — AI-Powered Fake News Detection System

**Validating and Exposing Real-world Information Through AI-driven Systems**

VERITAS is a web-based fake news detection system that analyzes news articles, URLs, and WhatsApp messages using Machine Learning and returns a verdict — REAL, FAKE, or UNCERTAIN — within seconds.

---

## 🚀 Featuress from **text, URLs, and WhatsApp messages**
- 🤖 **ML-powered** using TF-IDF + Logistic Regression (88-89% accuracy)
- 🔄 **Auto-retraining** every 6 hours using live RSS feeds (BBC, Reuters, AP, CNN)
- 📊 Live News Dashboard with real-time credibility scores
- 🔐 User authentication with secure hashed passwords
- 🧩 Chrome Browser Extension (Manifest V3)
- 📁 Analysis history saved per user
- ⚙️ Admin panel for model management

---

## 🛠️ Tech Stack
| Layer | Technologies |
|---|---|
| Backend | Python, Flask, SQLite |
| Machine Learning | Scikit-learn, TF-IDF, Logistic Regression, GridSearchCV |
| Frontend | HTML5, CSS3, JavaScript, Jinja2 |
| Other | BeautifulSoup, APScheduler, Joblib, Flask-Limiter |

---

## 📁 Project Structure
```
├── app.py               # Main Flask app & ML model
├── database.py          # SQLite functions
├── templates/           # HTML (Jinja2)
├── static/              # CSS, JS, Images
├── data/                # Training CSVs & database
├── ml_model/            # Saved model (.joblib)
└── browser_extension/   # Chrome extension
```

---

## 🎓 Final Year Project — BS Software Engineering# AI-Fake-News-Detection
AI-powered fake news detection system | Python, Flask, ML | FYP Project

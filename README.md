# Generative-and-Agentic-AI

## 📌 Automated Healthcare Diagnosis and Treatment Planner Recommendation System
	🚀 AI-powered healthcare diagnosis system that analyzes medical images, extracts symptoms, and provides accurate disease diagnosis and treatment plans using Computer Vision, NLP, and Generative AI.

### 🌟 Overview
	The Automated Healthcare Diagnosis and Treatment Planner Recommendation System is an AI-driven medical diagnosis tool designed to assist healthcare professionals in:
	✅ Analyzing medical images (X-rays, MRIs, CT scans, etc.)
	✅ Extracting symptoms and risk factors from medical reports
	✅ Generating an accurate diagnosis and treatment plan
	✅ Providing personalized recommendations for treatment and prevention
	It integrates Computer Vision (CV), Natural Language Processing (NLP), and Generative AI to deliver precise medical insights.

### ⚙️ Key Features

#### 🏥 Medical Image Analysis (CVAgent)
	Uses deep learning models to classify diseases from X-rays, MRIs, CT scans, ultrasounds, and histopathological images.
	Identifies affected organs, severity levels, abnormal biological changes, and possible causes.
 
#### 📝 Symptom Extraction from Reports (NLPAgent)
	Extracts symptoms, affected organs, severity levels, and risk factors from medical image reports.
	Helps understand biological impact and disease progression.

#### 🧑‍⚕️ AI-Powered Diagnosis & Treatment Plan (DiagnosisGenerator)
	Confirms disease based on extracted symptoms and image classification results.
	Provides medications, recommended tests, lifestyle changes, and prevention tips.

#### 💾 Caching System (CacheManager)
	Saves previous diagnosis results for faster processing.

#### 📜 Logging System (LoggerManager)
	Maintains detailed logs for debugging and system monitoring.
 
## 🏗️ Project Directory Structure
	healthcare_diagnosis_system/
	│── healthcare_diagnosis_system/
	│   ├── __init__.py
	│   ├── models/
	│   │   ├── __init__.py
	│   │   ├── medical_image.py
	│   │   ├── patient_record.py
	│   │   ├── diagnosis_report.py
	│   ├── agents/
	│   │   ├── __init__.py
	│   │   ├── cv_agent.py
	│   │   ├── nlp_agent.py
	│   │   ├── diagnosis_generator.py
	│   ├── workflows/
	│   │   ├── __init__.py
	│   │   ├── healthcare_diagnosis_workflow.py
	│   ├── utils/
	│   │   ├── __init__.py
	│   │   ├── cache.py
	│   │   ├── logger.py
	│   ├── main.py
	│── setup.py
	│── requirements.txt
	│── README.md
	
## 🚀 Installation Guide

  ### 📌 Prerequisites
 	Make sure you have the following installed:
		Python >=3.8
		pip (Python package manager)
		Virtual environment (venv or conda)
	
 ### 📥 Clone the Repository
	git clone https://github.com/dgx781/Generative-and-Agentic-AI/tree/DataScience-Ocean/health_diagnosis_recommender.git
	cd healthcare-diagnosis-recommender

 ### 🔹 Create a Virtual Environment & Activate it
	For Windows
 		python -m venv venv
		venv\Scripts\activate
	
	For Linux/macOS
 		python3 -m venv venv
		source venv/bin/activate
	
 ### 📦 Install Dependencies
	pip install -r requirements.txt
 
## 🏃 How to Run the System

 ### Step 1: Start the Application
	python main.py

 ### Step 2: Upload Medical Images
 	The system processes X-ray, MRI, CT scan, and histopathology images.
	It identifies diseases and their severity.

 ### Step 3: Extract Symptoms from Reports
 	The NLPAgent extracts symptoms and affected organs from text-based medical reports.

 ### Step 4: Generate Diagnosis & Treatment Plan
 	The DiagnosisGenerator confirms the disease, provides a structured diagnosis, and recommends treatments.

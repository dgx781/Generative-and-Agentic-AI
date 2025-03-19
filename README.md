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
	git clone https://github.com/dgx781/Generative-and-Agentic-AI.git

 ### 📥 Move to the health_diagnosis_recommender folder
 	cd health_diagnosis_recommender/

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

You need an API Key from Groq API to run the application. To get the API token, create a account in the groq API from the following link :- 

	https://console.groq.com/playground

To know more about groq  visit the website :-

	https://groq.com/

 ### Step 1: Install the package
 
 After cloning the repo, install the package with the help of pip :-
 
 	pip install health-diagnosis-recommender

 ### Step 2: Setup the GROQ_API_KEY either by directly passing the api_key as a parameter or by setting up the environment variable like :- 
 
 In the .env file :- 
 
	GROQ_API_TOKEN = "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"

 	GROQ_API_KEY = os.environ["GROQ_API_KEY]= GROQ_API_TOKEN 

 or directly passing the api_key as a parameter :- 

 	pip install groq

	import groq
	
	groq.Groq(api_key= "YOUR_API_KEY")
	 
 ### Step 3: Store the image path in a variable that would be later used for running the report
 
 	image_path = "Your image file path"

 ### Step 4: Create a table name and a session id for the Workflow Agent for each session encountered in the form of string which can be anything :- 
 	session_id : str = "Your session id in string format"
  	table_name : str = "Name of the table where the results will be stored in a tabular format"

 ### Step 5 Create a Workflow Storage object from the following code snippet given :- 
	from phi.storage.workflow.sqlite import SqlWorkflowStorage
	from health_diagnosis_recommender.main import HealthcareDiagnosisAgent
	from phi.utils.pprint import pprint_run_response
	from dotenv import load_dotenv
	import os
	
	load_dotenv()
	
	os.environ['GROQ_API_KEY'] = "YOUR API KEY"
	
	image_id = r"YOUR IMAGE PATH"
	
	agent = HealthcareDiagnosisAgent(
	    session_id= f"{ANY STR}", 
	    storage= SqlWorkflowStorage(
	        table_name= "YOUR TABLE NAME",
	        db_file= "ANY STR WITH.db extension"
	    )
	)
	
	results = agent.run(image_path= image_id, use_cache= True)
	
	pprint_run_response(results, markdown= True)
 
 where SqlWorkflowStorage is a inbuilt class of the phidata Agentic AI framework that needs two variables to be initialized , i.e., table name and db_file 

 ### Step 6 Run the above piece of code by replacing the placeholders with the information in your own file and data in the following code :- 
 
 	results = agent.run(image_path= image_id, use_cache= True)
	pprint_run_response(results, markdown= True)

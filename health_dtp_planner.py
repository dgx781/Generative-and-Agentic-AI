import json
import os
import base64
from typing import Optional, List, Iterator
from phi.utils.log import logger
from phi.agent import agent
from phi.model.groq import groq
from phi.workflow.workflow import Workflow
from phi.run.response import RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.pprint import pprint_run_response
from pydantic import BaseModel, Field
from typing import Iterator

    
class MedicalImage(BaseModel):
    path: str = Field(description="Path to the medical image file")
    image: Optional[str] = Field(description="Base64 encoded image data")
    model_config = {
        "arbitrary_types_allowed": True  
    }
    
    @classmethod
    def from_image(cls, path: str):
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return cls(path=path, image=encoded_string)
    
class PatientRecord(BaseModel):
    name : str = Field(description="Name of the patient")
    age : int = Field(description="Age of the patient")
    gender : str = Field(description="Gender of the patient")
    symptomps : str = Field(description="Symptoms of the patient")
    
class DiagniosisReport(BaseModel):
    disease_detected : str = Field(description="Disease detected in the patient")
    symptomps_captured: str = Field(description="Symptoms captured from the patient")
    organs_affected : str = Field(description="Organs affected by the disease")
    biological_changes : str = Field(description="Biological changes observed")
    internal_working_mechanism_functionality_impact : str = Field(description="Internal working mechanism and functionality impact")
    severity_level : str = Field(description="Severity level of the disease")
    harmul_disease_constituents : str = Field(description="Harmul disease constituents")
    risk_factor_working_mechanisms : str = Field(description="Risk factor working mechanisms")
    diagnosis : str = Field(description="Diagnosis of the patient")
    tests_performed : str = Field(description="Tests performed on the patient")
    medications_used : str = Field(description="Medication for the patient")
    hormonal_level_measures : List[int] = Field(description="Hormonal level measures for the patient")
    treatment : str = Field(description="Treatment of the patient")
    preventive_measures : str = Field(description="Preventive measures for the patient")
    lifestyle_changes : str = Field(description="Lifestyle changes for the patient")
    additional_info : Optional[str] = Field(description="Additional information of the patient")
    

class HealthcareDiagnosisAgent(Workflow):
    cv_agent : agent.Agent = agent.Agent(
        name= "Image Classifier",
        model = groq.Groq(id = "llama-3.2-11b-vision-preview"),
        description= "You are an advanced Medical Image Classifier Agent. Your role is to analyze and classify medical images, identifying potential diseases, abnormalities, and medical conditions. You process X-rays, CT scans, MRIs, ultrasounds, and histopathological images, providing a detailed classification of the detected conditions. Additionally, you provide an explanation of the detected disease, highlighting the affected organs, possible symptoms, and risk factors. You use deep learning-based vision models to extract insights from the images, focusing on: Disease classification (e.g., Pneumonia, Tumors, Fractures, Cardiovascular issues), Affected organs and tissues (e.g., Lungs, Liver, Brain, Heart), Severity levels (e.g., Mild, Moderate, Severe)., Potential causes (e.g., Genetic, Lifestyle, Infection-based)., Abnormal biological patterns (e.g., Tumor growth, Inflammation, Tissue damage)., Your output serves as crucial input for the NLP agent, which further extracts text-based symptoms, risk factors, and treatment options.",
        
        instructions= ["Load the provided medical image and perform pre-processing if necessary (e.g., grayscale conversion, contrast enhancement .Analyze the image using deep learning models to detect and classify medical conditions. Identify potential diseases or abnormalities present in the image Provide a detailed label and explanation for the detected disease, including:, Name of the disease, Affected organs, Severity level, Possible causes, Potential symptoms, Risk factors, If no disease is detected, clearly state No abnormalities found. but still highlight potential areas of interest., Ensure that the output is structured and clear, as it will be used by the NLP agent for further processing."],
        
        show_tool_calls= True,
        output_model= MedicalImage,
        markdown= True
    )
    
    nlp_agent : agent.Agent = agent.Agent(
        name= "Symptomp Extractor",
        model = groq.Groq(id = "llama-3.2-11b-vision-preview"),
        
        description= "You are an advanced Medical NLP Agent specializing in analyzing medical image reports and extracting crucial diagnostic details. Your role is to process medical image classifications (provided by the CV Agent) and extract relevant symptoms, affected organs, risk factors, severity levels, biological changes, and potential causes of the detected condition. Your analysis will help in: Identifying key symptoms associated with the detected condition, Assessing severity levels based on the detected abnormality, Determining affected organs and predicting functional impairment, Understanding biological changes caused by the disease, Recognizing potential risk factors (genetic, lifestyle, environmental), Describing how the disease impacts the internal working mechanism of the body, Your insights will be passed to the Diagnosis Generator Agent (gen_ai_agent), which will generate a comprehensive diagnosis and treatment plan for the patient.",
        
        instructions= ["Analyze the medical image classification results provided by the cv_agent, Extract and list key symptoms associated with the detected disease, Identify affected organs and explain how the disease impairs their function, Determine the severity level (Mild, Moderate, Severe, Critical), Examine and list biological changes (e.g., cell mutation, tissue damage, inflammation), Identify risk factors (e.g., genetics, lifestyle, environmental exposure) that may have contributed to the disease, Explain the internal working mechanism impact, describing how the disease disrupts normal bodily functions, Ensure the output is structured and well-organized, as it will be used by the Diagnosis Generator Agent to provide treatment recommendations,If no symptoms are identified, state No significant symptoms detected, but provide any relevant observations"],
        
        show_tool_calls= True,
        markdown= True
    )
    
    gen_ai_agent : agent.Agent = agent.Agent(
        name= "Diagnosis Generator",
        model = groq.Groq(id = "llama-3.2-11b-vision-preview"),
        
        description= "You are an AI-powered Medical Diagnosis and Treatment Generator Agent specializing in analyzing patient records, symptoms, and medical images to provide an accurate diagnosis, treatment plan, and preventive measures.Your role is to process extracted symptom data, disease classification, and affected organs (from the cv_agent and nlp_agent) to generate a comprehensive diagnosis report, including: Confirmed disease and its severity level, Symptoms and how they align with the detected condition, Affected organs and biological changes, Impact on internal bodily mechanisms, Harmful disease constituents and their effects, Risk factor mechanisms and progression risks, Diagnosis summary and test recommendations, Medication, treatments, and alternative therapies, Preventive measures and lifestyle changes. Your goal is to provide a medically accurate, structured, and informative diagnosis that aids healthcare professionals in decision-making and patient treatment planning.",
        
        instructions= ["Analyze medical image classification and symptom extraction results from cv_agent and nlp_agent. Identify and confirm the detected disease based on symptom patterns and organ impact. Determine the severity level of the disease (Mild, Moderate, Severe, Critical).Provide a detailed diagnosis covering: Disease name and type (e.g., infectious, autoimmune, genetic), Affected organs and biological changes, Internal functionality impairment and disease progression risks, Possible harmful constituents of the disease, List recommended medical tests (e.g., MRI, blood tests, biopsies) for further evaluation, Suggest appropriate medications (including drug types and dosages where applicable), Outline treatment options (surgical, therapeutic, lifestyle-based), Provide preventive measures to slow or stop disease progression, Recommend lifestyle changes (e.g., diet, exercise, stress management) to improve patient well-being. Ensure that the diagnosis report is detailed, structured, and medically relevant, formatted for easy interpretation by healthcare professionals."],
        
        show_tool_calls= True,
        markdown= True
    )
    
    def run(self, image_path: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(f"Processing image: {image_path}")
        
        if use_cache:
            cached_cv_response = self.cached(image_path)
            if cached_cv_response:
                logger.info("Found a cached label for the image")
                yield RunResponse(
                    content= cached_cv_response,
                    event= RunEvent.workflow_completed,
                )
                return
            
        results: Optional[MedicalImage] = self.get_disease_label(image_path)
        if results is None:
            yield RunResponse(
                event= RunEvent.workflow_completed,
                response= "No CV data found"
            )
            return
        
        symptoms : Optional[PatientRecord] = self.extract_symptomps(image_path, results)
        
        if symptoms is None:
            logger.info("No symptoms found")
            yield RunResponse(
                event= RunEvent.workflow_completed,
                response= "No diagnosis found"
            )
            return
        
        for response in self.provide_preventive_measures_And_remedials(image_path, results, symptoms):
            yield response
        
        
    def get_disease_label(self, image_path: str) -> Optional[MedicalImage]:
        logger.info("Identifying the disease in the image")
        results = self.cv_agent.run(image_path)
        if results is None:
            raise ValueError("No CV data found")
        self.session_state.setdefault("labels", {})[image_path] = results
        return self.session_state["labels"][image_path]
    
    def cached(self, image_path: str)-> Optional[MedicalImage]:
        logger.info("Checking cache for existing label")
        return self.session_state.get("labels", {}).get(image_path)
    
    def extract_symptomps(self, image_path: str, results: MedicalImage)-> PatientRecord:
        logger.info("Describe the disease and extracting symptomps from the image")
        results = self.nlp_agent.run(image_path)
        self.session_state.setdefault("symptomps", {})[image_path] = results
        return results
        
    def recomment_treatments(self, image_path: str, results: MedicalImage,  symptomps: PatientRecord)-> DiagniosisReport:
        logger.info("Recommendng treatments for the patient")
        agent_input = {
        "image_path": image_path,
        "results": results.model_dump(),  
        "symptomps": symptomps.model_dump() 
    }
        recommendations: RunResponse = self.gen_ai_agent.run(json.dumps(agent_input, indent=4))

        if not recommendations or not recommendations.content:
            raise ValueError("No recommendations found")

        self.session_state.setdefault("recommendations", {})[image_path] = recommendations.content
        return recommendations.content
    
    def add_response_to_cache(self, image_path: str, response: DiagniosisReport):
        logger.info(f"Caching response for image: {image_path}")
        self.session_state.setdefault("recommendations", {})[image_path] = response
    
    def provide_preventive_measures_And_remedials(self, image_path: MedicalImage, results: MedicalImage, symptomps: PatientRecord)-> Iterator[RunResponse]:
        logger.info("Providing preventive measures and remedies for the patient")
        provider_inputs = {
            "image_path" : image_path,
            "image_analysis" : results.model_dump(),
            "symptomps" : symptomps.model_dump(),
        }
        response = self.gen_ai_agent.run(json.dumps(provider_inputs, indent = 4))
        
        self.add_response_to_cache(image_path, response)
        
        yield RunResponse(content=response.content, event=RunEvent.workflow_completed)
        

if __name__ == "__main__":
    from rich.prompt import Prompt
    file_path = Prompt.ask("[bold]Enter the path to your local file[/bold]\n📂")

    if not os.path.exists(file_path):
        print("❌ Error: The file path does not exist. Please enter a valid path.")
        exit()

    summarizer = HealthcareDiagnosisAgent(
        session_id=f"Image Diagnosis -{file_path.replace(':', '').replace('\\', '_').replace('\\', '_')}",
        storage=SqlWorkflowStorage(
            table_name="Image Diagnosis Results",
            db_file="tmp/workflows.db",
        ),
    )

    # Run the summarizer with the local file
    summary: Iterator[RunResponse] = summarizer.run(image_path=file_path, use_cache=True)
    pprint_run_response(summary, markdown=True)
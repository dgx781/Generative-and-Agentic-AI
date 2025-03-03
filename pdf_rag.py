from typing import Optional , List
import typer
from phi.llm.groq import groq
from phi.assistant.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.arxiv import ArxivKnowledgeBase
from phi.vectordb.pgvector.pgvector2 import PgVector2
import os
from dotenv import load_dotenv

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

kb = ArxivKnowledgeBase(
    urls= ["https://arxiv.org/pdf/2501.00089"],
    vector_db= PgVector2(collection= "Astrophysics", db_url= db_url),
)

kb.load()

store = PgAssistantStorage(table_name= "arxiv_assist", db_url= db_url)

def arxiv_assist(new : bool = False, user : str = "user"):
    run_id: Optional[str] = None
    
    if not new:
        existing_run_ids : List[str] = store.get_all_run_ids(user)
        if len(existing_run_ids)> 0:
            run_id = existing_run_ids[0]
            
    assistant = Assistant(
        name= "arxiv_assist",
        role= "You are an arxiv assistant. You are an expert in astrophysics. You can answer questions about astrophysics.",
        instructions= ["Include a detailed elaborate response mentioning all the details about the formats of data mentioned such as images, videos, audio and text"],
        llm= groq.Groq(id= "llama-3.2-11b-vision-preview"),
        knowledge_base= kb,
        storage= store,
        search_knowledge= True,
        show_tool_calls= True,
        read_chat_history= True,
        run_id= run_id,
        user_id= user,
    )
    
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started running : {run_id}\n")
        
    else:
        print(f"Continuing running : {run_id}\n")
        
    assistant.cli_app(markdown= True)
    

if __name__ == "__main__":
    typer.run(arxiv_assist)
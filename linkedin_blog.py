from typing import Optional , List, Iterator
import json
from pydantic import BaseModel, Field
from phi.agent import agent
from phi.model.groq import groq
from phi.workflow.workflow import Workflow
from phi.run.response import RunResponse , RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger
import os
from dotenv import load_dotenv

load_dotenv("C:/Users/DIBYOJIT/Videos/ds_ca2/.env")

class LinkedinProfile(BaseModel):
    profile_url: str = Field(..., description="The URL of the LinkedIn profile")
    name: str = Field(..., description="The full name of the LinkedIn user")
    job_title: Optional[str] = Field(None, description="The current job title of the user")
    company: Optional[str] = Field(None, description="The current company the user works for")


class Linkedin(BaseModel):
    url: str = Field(..., description="The URL of the Linkedin post")
    post : Optional[str] = Field(..., description="The content of the Linkedin post")
    
class SearchResults(BaseModel):
    profile: LinkedinProfile = Field(..., description="The LinkedIn profile details")
    posts: List[Linkedin] = Field(..., description="The list of Linkedin posts")
    
class LinkedinSummarize(Workflow):
    fetch : agent.Agent = agent.Agent(
        name= "Crawler",
        model = groq.Groq(id = "llama-3.2-11b-vision-preview"),
        role= "You are a web crawler agent. You can search for information on the internet and provide the results to the user.",
        instructions= ["Extract the full name, job title, and company from the given LinkedIn profile URL.",
            "Retrieve all public LinkedIn posts from this profile.",
            "If no posts are found, return 'No posts available'."],
        description= "You will extract LinkedIn profile details and posts.",
        show_tool_calls= True,
        output_model= SearchResults,
        markdown= True
    )
    
    summarizer : agent.Agent = agent.Agent(
        name= "Summarizer",
        model = groq.Groq(id = "llama-3.2-11b-vision-preview"),
        role= "You are a summarizer agent for Linkedin Posts. Summarize the posts posted by the user with all the elaborate details mentioned in the post",
        instructions= ["Analyze the user's LinkedIn posts and provide recommendations.",
            "Suggest potential topics the user should post about based on trends in their industry.",
            "If the user is in a specific field, suggest professional networking or skill improvements."],
        description= "You provide content and career recommendations based on LinkedIn activity.",
        show_tool_calls= True,
        markdown= True
    )
    
    def run(self,  profile_url: str, use_cache: bool = True)-> Iterator[RunResponse]:
        logger.info(f"Fetching LinkedIn profile and posts from: {profile_url}")
        
        if use_cache:
            cached_linkedin_response = self.cached(profile_url)
            if cached_linkedin_response:
                logger.info("Found a cached summary for the topic")
                yield RunResponse(
                    content= cached_linkedin_response,
                    event= RunEvent.workflow_completed,
                )
                return
            
        results: Optional[SearchResults] = self.get_profile_and_posts(profile_url)
        if results is None:
            yield RunResponse(
                event= RunEvent.workflow_completed,
                response= "No LinkedIn data found"
            )
            return
        
        yield from self.recommend_content(results)
        
    def cached(self : str , header : str)-> Optional[str]:
        logger.info("Checking cache for existing summary")
        return self.session_state.get("summaries", {}).get(header)
    
    def add_summary_to_cache(self, profile_url : str , summary : Optional[str]):
        logger.info(f"Caching summary for topic: {profile_url}")
        self.session_state.setdefault("summaries", {})
        self.session_state["summaries"][profile_url] = summary
        
    def get_profile_and_posts(self, profile_url: str) -> Optional[SearchResults]:
        try:
            searcher_response: RunResponse = self.fetch.run(profile_url)
            if not searcher_response or not searcher_response.content:
                logger.warning("Empty response from fetcher")
                return None
            return searcher_response.content
        except Exception as e:
            logger.error(f"Error fetching LinkedIn posts: {str(e)}")
            return None

    def recommend_content(self, search_results: SearchResults) -> Iterator[RunResponse]:
        logger.info("Generating recommendations based on LinkedIn posts")
        recommender_input = {
            "profile": search_results.profile.model_dump(),
            "posts": [p.model_dump() for p in search_results.posts]
        }
        response= self.summarizer.run(json.dumps(recommender_input, indent=4))
        
        self.add_summary_to_cache(search_results.profile.profile_url, response.content)
        
        yield RunResponse(content=response.content, event=RunEvent.workflow_completed)

if __name__ == "__main__":
    from rich.prompt import Prompt
    
    profile_url = Prompt.ask("[bold]Enter the LinkedIn Profile URL[/bold]\n🔗")

    summarizer = LinkedinSummarize(
        session_id=f"linkedin-summary-{profile_url.replace('https://', '').replace('/', '_')}",
        storage=SqlWorkflowStorage(
            table_name="linkedin_summaries",
            db_file="tmp/workflows.db",
        ),
    )

    summary: Iterator[RunResponse] = summarizer.run(profile_url = profile_url , use_cache=True)
    pprint_run_response(summary, markdown=True)

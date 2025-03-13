from agno.agent import agent
from agno.models.groq import groq
from agno.models.cohere.chat import Cohere
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['CO_API_KEY'] = os.getenv("CO_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

web_agent = agent.Agent(
    name= "WWW Crawler",
    model= groq.Groq(id ="llama-3.3-70b-versatile"),
    description= "Search the WWW for relevant information regarding the users questions and queries and provide a detailed elaborate description",
    instructions= ["Search for the key parameters and retrieve the essential context that the user wants to specify and regarding those key parameters and contexts provide a detailed explanation of the response"],
    tools= [DuckDuckGoTools()],
    show_tool_calls= True,
    markdown= True
)

fin_agent = agent.Agent(
    name= "Financial Analyst",
    model= Cohere(),
    tools= [YFinanceTools(stock_price= True, analyst_recommendations= True, stock_fundamentals= True, company_news= True, income_statements= True)],
    description= "Provide the financial information for the user query in a illustrative and tabular format. If you don't find any results, say 'No results found'.",
    instructions= "Search for the financial news. stock prices, key parameters affecting the stock prices, Other stick prices comparisons, investment profits and losses key competetors and related information in a tabular and descriptive elaborate format",
    markdown= True,
    show_tool_calls= True
)

agent_team = agent.Agent(
    team= [web_agent, fin_agent],
    model= groq.Groq(id ="llama-3.3-70b-versatile"),
    instructions= ["Search the WWW for relevant information regarding the users questions and queries and provide a detailed elaborate description. Search for the key parameters and retrieve the essential context that the user wants to specify and regarding those key parameters and contexts provide a detailed explanation of the response", "Provide the financial information for the user query in a illustrative and tabular format. If you don't find any results, say 'No results found'. Search for the financial news. stock prices, key parameters affecting the stock prices, Other stick prices comparisons, investment profits and losses key competetors and related information in a tabular and descriptive elaborate format"],
    markdown= True,
    show_tool_calls= True
)

agent_team.print_response("Provide a detailed financial analysis of the comparison of stock prices of apple, tesla, nvidia and Google based on detailed statistical analysis ans statistical methods and provide an Indepth Elaborate description of their Financial Analysis Report and suggest which one would be the best option to invest for and describe in detail the justification for your following response. ", stream= True)
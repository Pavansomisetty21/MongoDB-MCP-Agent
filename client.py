import os
import sys
import asyncio
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from mcp import StdioServerParameters,ClientSession
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_groq import ChatGroq

import requests
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.environ.get("GROQ_KEY"))
server_parameters = StdioServerParameters(
    command=sys.executable,
    args=['server.py']
)

memory = MemorySaver()

async def main():
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(
                model,
                tools,
                prompt=(
                    "You are a MongoDB assistant. "
                    "Your job is to use the tools from the MCP server to interact with MongoDB safely and effectively. "
                    "Use tools to fetch schema info, update documents, disable jobs via config collections, and more. "
                    "Never guess. Use the tools provided for every action or question about data."
                   "provide accurate data from the database
                ),
                checkpointer=memory,
            )
            
            config = {"configurable": {"thread_id": "335885"}}

            while True:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Exiting...")
                    break

                agent_response = await agent.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )
                print("Bot:", agent_response["messages"][-1].content)

if __name__ == "__main__":
   asyncio.run(main())

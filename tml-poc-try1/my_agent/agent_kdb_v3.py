#!/usr/bin/env python3

import os
from strands import Agent
from strands.models import BedrockModel
from strands_tools import use_aws
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters, stdio_client
from prereq_chromadb_setup import retrieve_chunks, save_to_chroma

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AWS_REGION = "ap-south-1"

AGENT_CONFIG = {
    "system_prompt": """
    You are an expert AWS Cloud Engineer assistant...
    -Be concise and factual.
    -Provide step-by-step AWS CLI or console instructions.
    -Avoid unnecessary explanations unless asked.
    -Use markdown for code blocks.
    -Do not hallucinate AWS services.
    """
}

MCP_SERVERS = [
    {
        "name": "AWS Documentation MCP Server",
        "command": "uvx",
        "args": ["awslabs.aws-documentation-mcp-server@latest"]
    },
    {
        "name": "AWS Diagram MCP Server",
        "command": "uvx",
        "args": ["awslabs.aws-diagram-mcp-server@latest"]
    },
    {
        "name": "AWS Pricing & Cost Analysis MCP Server",
        "command": "uvx",
        "args":["awslabs.aws-pricing-mcp-server@latest"]
    }
#    {
#        "name": "AWS Troubleshooting MCP Server",
#        "command": "uvx",
#        "args":["awslabs.aws-troubleshooting@latest"]
#    }

#    {
#        "name": "AWS Security & Compliance MCP Server",
#        "command": "uvx",
#        "args":["awslabs.aws-security-mcp-server@latest"]
#    }

]

BEDROCK_MODEL_CONFIG = {
    "model_id": "arn:aws:bedrock:ap-south-1:099064381286:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0",
    "region_name": AWS_REGION,
    "temperature": 0.2,
    "max_tokens": 2000,
    "top_p": 0.7
}


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
def configure_environment():
    """Set required environment variables."""
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION


# ---------------------------------------------------------------------------
# MCP Client Setup
# ---------------------------------------------------------------------------
def start_mcp_clients():
    """
    Start all MCP clients as per MCP_SERVERS configuration.
    Returns a combined list of all tools.
    """
    env_copy = os.environ.copy()
    env_copy["AWS_DEFAULT_REGION"] = AWS_REGION

    all_tools = []
    for server in MCP_SERVERS:
        client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=server["command"],
                args=server["args"],
                env=env_copy
            )
        ))
        print(f"Starting MCP Client: {server['name']}")
        client.start()
        tools = client.list_tools_sync()
        all_tools.extend(tools)

    return all_tools


# ---------------------------------------------------------------------------
# Agent Creation
# ---------------------------------------------------------------------------
def create_agent():
    """Initialize and return the AWS Cloud Agent."""
    bedrock_model = BedrockModel(**BEDROCK_MODEL_CONFIG)
    mcp_tools = start_mcp_clients()

    return Agent(
        tools=[use_aws] + mcp_tools,
        model=bedrock_model,
        system_prompt=AGENT_CONFIG["system_prompt"]
    )


# ---------------------------------------------------------------------------
# Main Interaction Loop
# ---------------------------------------------------------------------------
def run_agent(agent_instance):
    """Interactive loop with retrieval + Claude + MCP + persistence"""
    while True:
        user_input = input("\nEnter your query [or type 'exit']: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!!!\n\nRegards,\nAWSMate\n")
            break

        # Step 1: Check Chroma DB
        retrieved = retrieve_chunks(user_input, top_k=2)
        context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i,c in enumerate(retrieved)]) if retrieved else ""

        # Step 2: If context exists, prepend it
        if context_text:
            enriched_query = f"Use the following context if relevant:\n{context_text}\n\nUser Question: {user_input}"
        else:
            enriched_query = user_input

        print("\n" + "="*100)
        print(">>> Agent's Response <<<".center(100))
        print("="*100)

        # Step 3: Run through Strands Agent (Claude + MCP)
        response = agent_instance(enriched_query)
        if isinstance(response, str):
            print(response)

            # Step 4: Save new knowledge into Chroma
            save_to_chroma(user_input, response, source="Claude+MCP")

        print("\n" + "="*100)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_environment()
    aws_agent = create_agent()
    run_agent(aws_agent)


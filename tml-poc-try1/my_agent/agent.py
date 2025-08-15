from strands import Agent
from strands.models import BedrockModel
from strands_tools import use_aws
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters, stdio_client
import os, boto3


my_env = os.environ.copy()
my_env["AWS_DEFAULT_REGION"] = "ap-south-1"
os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
#session = boto3.Session(region_name='ap-south-1')
#os.system("aws configure set region ap-south-1 --profile default")
#aws configure set region ap-south-1 --profile default

# Create a model instance
bedrock_model = BedrockModel(
        #model_id="anthropic.claude-3-haiku-20240307-v1:0",
        #region_name="ap-south-1",
        model_id="arn:aws:bedrock:ap-south-1:680918385129:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0",  
        region_name="ap-south-1",
    temperature=0.1,
)


# Set up AWS Documentation MCP client
aws_docs_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"], env=my_env)
))
aws_docs_mcp_client.start()

# Set up AWS Diagram MCP client
aws_diagram_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-diagram-mcp-server@latest"], env=my_env)
))
aws_diagram_mcp_client.start()


# Get tools from MCP clients
docs_tools = aws_docs_mcp_client.list_tools_sync()
diagram_tools = aws_diagram_mcp_client.list_tools_sync()


# Add all tools to the agent
agent = Agent(
    tools=[use_aws] + docs_tools + diagram_tools,
    model=bedrock_model,
    system_prompt="You are an expert AWS Cloud Engineer assistant..."
)


'''
# Create the agent with tools and model
agent = Agent(
    tools=[use_aws],  # AWS CLI tool
    model=bedrock_model,
    system_prompt="You are an expert AWS Cloud Engineer assistant..."
)
'''



# Execute a task

while True:
    user_input = input("\nEnter your query [or type 'exit' to quit] : ").strip()
    print("\n\n$$$$$$#########********<<<<<<*>>>>>> Agent's Response <<<<<<*>>>>>>********#########$$$$$$\n")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("\nGoodbye!!!\n\nRegards,\nAWSMate...\n")
        break
    print("\n#########################################################################################\n")
    
    response = agent(user_input)
    #print("\n\n$$$$$$#########********<<<>>>>>> Agent's Response <<<<<<>>>********#########$$$$$$\n")
    #print(response)
    print("\n#########################################################################################\n")

# response = agent(input("Enter your query : "))

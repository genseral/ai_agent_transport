# AI Agent Transport

This repository contains the implementation of a tool-calling agent designed to assist users in planning their travel by querying transportation APIs for train connections, station boards, and real-time updates. The agent is built using the Mosaic AI Agent Framework and integrates with Databricks and LangChain.

## Repository Structure

- `export_full_agent_implementation.py` - Main driver notebook auto-generated from Databricks AI Playground export
- `travel_agent.py` - Core agent implementation file
- `system_prompt.txt` - System prompt for the agent defining its behavior and capabilities
- `function_implementations.ipynb` - Notebook containing the Unity Catalog function implementations for train connections and station boards
- `getting_started_agent_implementation.ipynb` - Example notebook showing basic agent usage

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Agent Logic](#agent-logic)
- [Testing the Agent](#testing-the-agent)
- [Logging and Deployment](#logging-and-deployment)
- [Evaluation](#evaluation)
- [Pre-deployment Validation](#pre-deployment-validation)
- [Model Registration and Deployment](#model-registration-and-deployment)
- [Next Steps](#next-steps)
- [API Reference](#api-reference)

## Prerequisites

- Databricks account
- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

## Installation

To install the required Python packages, run the following command:

```bash
%pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv
```

## Usage

### System Prompt

The agent requires a system prompt that defines its behavior, capabilities, and limitations. This prompt is provided in the `system_prompt.txt` file and is automatically loaded by the agent implementation. The system prompt defines:

- Purpose and limitations
- Available parameters
- Data sources and actions
- Error handling behavior
- Sample questions and scenarios

### Define the Agent

The agent is defined in the `travel_agent.py` file. It uses the `ChatDatabricks` model and integrates with Unity Catalog functions to retrieve train connections and station boards. The core functionality is implemented through two main functions:

- `get_connections` - Retrieves train connections between stations
- `get_station_board` - Gets departure/arrival information for a specific station

These functions are implemented in the `function_implementations.ipynb` notebook and are registered in Unity Catalog for use by the agent.

### Agent Logic

The agent logic is implemented in the `create_tool_calling_agent` function, which binds the language model to the tools and defines the workflow for handling user messages and tool calls.

### Testing the Agent

You can test the agent by running the following commands in a Databricks notebook:

```python
from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Hello!"}]})
```

### Logging and Deployment

The agent can be logged as an MLflow model and deployed using the following commands:

```python
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "find the next train from zurich to bern?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "langchain",
            "langgraph==0.3.4",
            "databricks-langchain",
            "unitycatalog-langchain[databricks]",
            "pydantic",
        ],
        resources=resources,
    )
```

### Evaluation

Evaluate the agent using the `mlflow.evaluate` API:

```python
import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "Find the next train from zurich to bern?"
                }
            ]
        },
        "expected_response": None
    },
    # Add more evaluation examples as needed
]

eval_dataset = pd.DataFrame(eval_examples)

import mlflow

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",
        data=eval_dataset,
        model_type="databricks-agent",
    )

display(eval_results.tables['eval_results'])
```

### Pre-deployment Validation

Validate the agent before deployment:

```python
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)
```

### Model Registration and Deployment

Register and deploy the model to Unity Catalog:

```python
mlflow.set_registry_uri("databricks-uc")

catalog = "travel_agents"
schema = "train_agent"
model_name = "train_travel_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags={"endpointSource": "playground"})
```

### Next Steps

After deploying the agent, you can interact with it in the AI playground, share it with SMEs for feedback, or embed it in a production application.

## API Reference

Link to the API used: [Transport Open Data](https://transport.opendata.ch/)

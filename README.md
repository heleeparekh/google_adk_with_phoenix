# ADK Walkthrough: Google ADK Agents with Arize Phoenix Tracing

This project demonstrates how to create and manage different agents using the [Google Agent Development Kit (ADK)](https://github.com/google/agent-development-kit) and integrate them with [Arize Phoenix](https://github.com/Arize-ai/phoenix) for tracing and observability.

## Features

- Build and configure multiple agents using Google ADK
- Integrate agents with Arize Phoenix for end-to-end tracing
- Example workflows and tracing visualization

## Getting Started

### Prerequisites

- Python 3.8+
- [Google ADK](https://google.github.io/adk-docs/)
- [Arize Phoenix](https://arize.com/docs/phoenix)

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/heleeparekh/google_adk_with_phoenix.git
    cd google_adk_with_phoenix
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage


1. **Start the Phoenix server** (required for tracing):
    ```sh
    phoenix serve
    ```
1. **Set required environment variables** for authentication and configuration.
2. **Run an agent script** of your choice to start an agent and enable tracing. For example:
    ```sh
    python coding_agent.py
    ```
    or
    ```sh
    python math_assistant.py
    ```
3. **View traces** in the Arize Phoenix dashboard.
   
<img width="1429" height="691" alt="image" src="https://github.com/user-attachments/assets/137290bd-00c2-4a6f-8a2a-82a749b5e05c" />

<img width="1303" height="780" alt="image" src="https://github.com/user-attachments/assets/327c5f8e-dbe8-4fa8-8516-ed0a967fb69f" />


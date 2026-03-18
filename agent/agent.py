"""LangChain agent with ClinVar tool and ACMG classification."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent.acmg import classify_variant, evaluate_acmg_criteria
from agent.clinvar import query_clinvar

load_dotenv()

SYSTEM_PROMPT = """You are a clinical molecular geneticist specializing in \
hereditary cancer risk assessment. Your role is to classify germline variants \
using the ACMG/AMP 2015 guidelines.

When a user provides a variant, you should:
1. Query ClinVar using the clinvar_lookup tool to retrieve the variant record.
2. Analyze the ClinVar data and apply ACMG criteria evaluation.
3. Provide a structured classification with reasoning.

Always be precise and evidence-based. Cite the specific ACMG criteria that \
apply and explain why. If data is limited, acknowledge the uncertainty.

IMPORTANT: You must always use the clinvar_lookup tool to look up variants. \
Do not rely on your training data for variant classifications, as ClinVar \
is updated regularly and your knowledge may be outdated.

After retrieving ClinVar data, provide a plain-English summary of your \
classification reasoning, including which ACMG criteria apply and why."""

DISCLAIMER = (
    "This is an AI-assisted research tool and should not be used for clinical "
    "decision-making. Variant classifications should be reviewed by a "
    "certified clinical molecular geneticist and confirmed through validated "
    "clinical-grade processes."
)


@tool
def clinvar_lookup(variant: str) -> str:
    """Query ClinVar for a genomic variant and return structured data.

    Use this tool to look up any genomic variant in the NCBI ClinVar database.
    Input should be a variant description such as 'BRCA1 c.5266dupC' or
    'NM_007294.4:c.5266dupC' or 'TP53 c.817C>T'.

    Returns a JSON string with variant details including clinical significance,
    review status, submitter information, and associated conditions.
    """
    result = query_clinvar(variant)
    return json.dumps(result, indent=2, default=str)


def build_agent():
    """Build and return the LangGraph react agent with ClinVar tools."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=0,
        max_tokens=4096,
    )

    tools = [clinvar_lookup]

    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    return agent


def classify_variant_with_agent(variant: str) -> dict[str, Any]:
    """Run the full classification pipeline for a variant.

    This function:
    1. Queries ClinVar via the agent
    2. Applies ACMG criteria evaluation
    3. Combines agent reasoning with programmatic classification

    Args:
        variant: Variant string, e.g. "BRCA1 c.5266dupC"

    Returns:
        Structured classification result dict.
    """
    # Step 1: Query ClinVar directly for programmatic ACMG evaluation
    clinvar_record = query_clinvar(variant)

    if clinvar_record.get("error") and not clinvar_record.get("variant_id"):
        return {
            "variant": variant,
            "clinvar_record": clinvar_record,
            "criteria_triggered": [],
            "classification": "Unable to classify",
            "confidence": "N/A",
            "reasoning": clinvar_record["error"],
            "disclaimer": DISCLAIMER,
        }

    # Step 2: Evaluate ACMG criteria programmatically
    criteria = evaluate_acmg_criteria(clinvar_record)

    # Step 3: Apply ACMG combination rules
    classification_result = classify_variant(criteria)

    # Step 4: Run the LangChain agent for enhanced reasoning
    agent = build_agent()
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=f"Classify the following variant: {variant}")]}
        )
        # Extract the last AI message content
        messages = response.get("messages", [])
        agent_reasoning = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                agent_reasoning = msg.content
                break
    except Exception as e:
        agent_reasoning = f"Agent reasoning unavailable: {str(e)}"

    # Step 5: Combine results
    enhanced_reasoning = classification_result["reasoning"]
    if agent_reasoning and "reasoning" not in agent_reasoning[:50].lower():
        enhanced_reasoning = (
            f"{classification_result['reasoning']}\n\n"
            f"Additional AI analysis: {agent_reasoning}"
        )

    return {
        "variant": variant,
        "clinvar_record": clinvar_record,
        "criteria_triggered": criteria,
        "classification": classification_result["classification"],
        "confidence": classification_result["confidence"],
        "reasoning": enhanced_reasoning,
        "disclaimer": DISCLAIMER,
    }

"""Deep research workflow exposed as FastAPI endpoints."""
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from agents import Runner, trace, gen_trace_id,Agent,OpenAIChatCompletionsModel,WebSearchTool,ModelSettings,function_tool
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep-research", tags=["Deep Research"])


class DeepResearchRequest(BaseModel):
    """Request payload for deep research."""
    query: str = Field(..., description="User supplied research query.")
    email: str = Field(..., description="Email address to send the report to.")
    workflow_models: dict[str, int] = Field(
        default_factory=lambda: {
            "search_plan": 1,
            "perform_search": 1,
            "write_report": 1,
            "send_email": 1
        },
        description="Mapping specifying model/variant selections for workflow stages."
    )
    search_count: int = Field(
        default=5,
        description="Number of web searches to perform."
    )


class ReportData(BaseModel):
    """Response containing the final research report."""
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")

    markdown_report: str = Field(description="The final report")

    follow_up_questions: list[str] = Field(description="Suggested topics to research further")

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")

    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


def model_selector(sln: int) -> Any:
    """
    Returns an OpenAIChatCompletionsModel instance based on the selected model number.

    Args:
        sln (int): Model selector number.
            1 -> LLaMA 3.2 3B (via Ollama)
            3 -> Qwen 3-30B (via OpenRouter)
            2 -> DeepSeek R1-LLaMA 70B (via OpenRouter)
            4 -> GPT-4o-mini (via OpenAI)
            Default -> LLaMA 3.2 3B (via Ollama)

    Returns:
        OpenAIChatCompletionsModel: Configured model instance.
    """

    # --- Load environment variables ---
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    openrouter_base_url = os.getenv("OPENAI_ROUTER_BASE_URL")
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # --- Validate environment ---
    missing_env = [
        var for var, val in {
            "OLLAMA_BASE_URL": ollama_base_url,
            "OPENAI_ROUTER_BASE_URL": openrouter_base_url,
            "OPEN_ROUTER_API_KEY": openrouter_api_key,
            "OPENAI_API_KEY": openai_api_key,
        }.items() if not val
    ]
    if missing_env:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_env)}")

    # --- Lazy client initialization ---
    ollama_client = AsyncOpenAI(base_url=ollama_base_url, api_key="ollama")
    openrouter_client = AsyncOpenAI(base_url=openrouter_base_url, api_key=openrouter_api_key)
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    # --- Define model registry ---
    models = {
        1: OpenAIChatCompletionsModel(
            model="llama3.2:3b",
            openai_client=ollama_client
        ),
        3: OpenAIChatCompletionsModel(
            model="qwen/qwen3-30b-a3b:free",
            openai_client=openrouter_client
        ),
        2: OpenAIChatCompletionsModel(
            model="deepseek/deepseek-r1-distill-llama-70b:free",
            openai_client=openrouter_client
        ),
        4: OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=openai_client
        ),
    }

    # --- Return selected model (default → Ollama LLaMA) ---
    return models.get(sln, models[1])


class ResearchManager:
    """Handles the multi-agent deep research workflow."""

    async def run(self, request: DeepResearchRequest) -> ReportData:
        """Run the deep research process, logging status updates and returning the final report."""
        
        # Different Agents can be selected based on workflow_models in request

        logging.info("DeepResearchRequest received by :", request.email, request)

        #####
        # PLANNER_AGENT
        #####   
        PLANNER_AGENT_INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of web searches \
        to perform to best answer the query. Output {request.search_count} terms to query for."
        planner_agent = Agent(
            name="PlannerAgent",
            instructions=PLANNER_AGENT_INSTRUCTIONS,
            model=model_selector(request.workflow_models.get("search_plan", 1)),
            output_type=WebSearchPlan,
        )
        
        #####
        # SEARCH_AGENT
        #####
        SEARCH_AGENT_INSTRUCTIONS = "You are a research assistant. Given a search term, you search the web for that term and \
        produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
        words. Capture the main points. Write succintly, no need to have complete sentences or good \
        grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
        essence and ignore any fluff. Do not include any additional commentary other than the summary itself."

        search_agent = Agent(
            name="Search agent",
            instructions=SEARCH_AGENT_INSTRUCTIONS,
            tools=[WebSearchTool(search_context_size="low")],
            model="gpt-4o-mini",
            model_settings=ModelSettings(tool_choice="required"),
        )
        
        
        #####
        # WRITER_AGENT
        #####
        WRITER_AGENT_INSTRUCTIONS = (
        "You are a senior researcher tasked with writing a cohesive report for a research query. "
        "You will be provided with the original query, and some initial research done by a research assistant.\n"
        "You should first come up with an outline for the report that describes the structure and "
        "flow of the report. Then, generate the report and return that as your final output.\n"
        "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
        "for 5-10 pages of content, at least 1000 words."
        )
        writer_agent = Agent(
        name="WriterAgent",
        instructions=WRITER_AGENT_INSTRUCTIONS,
        model=model_selector(request.workflow_models.get("write_report", 1)),
        output_type=ReportData,
      )
        #####
        # EMAIL_AGENT
        #####
        
        EMAIL_AGENT_INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
        You will be provided with a detailed report. You should use your tool to send one email, providing the 
        report converted into clean, well presented HTML with an appropriate subject line."""

        @function_tool
        def send_email(subject: str, html_body: str) -> Dict[str, str]:
            """Send an email using SendGrid with the given subject and HTML body."""
            SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

            if not SENDGRID_API_KEY:
                return {"status": "error", "message": "SENDGRID_API_KEY not found"}

            try:
                message = Mail(
                    from_email=os.getenv("FROM_EMAIL", "justdevaplaying@gmail.com"),
                    to_emails=os.getenv("TO_EMAIL", request.email),
                    subject=subject,
                    html_content=html_body,
                )

                sendgrid_client = SendGridAPIClient(SENDGRID_API_KEY)
                response = sendgrid_client.send(message)

                return {
                    "status": "success",
                    "code": response.status_code,
                }

            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        email_agent = Agent(
            name="Email agent",
            instructions=EMAIL_AGENT_INSTRUCTIONS,
            tools=[send_email],
            model=model_selector(request.workflow_models.get("send_email", 1)),
        )
        
        
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            trace_message = f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            logger.info(trace_message)

            logger.info("Starting research...")
            search_plan = await self.plan_searches(request.query, planner_agent)
            logger.info("Searches planned, starting to search...")

            search_results = await self.perform_searches(search_plan,search_agent)
            logger.info("Searches complete, writing report...")

            report = await self.write_report(request.query, search_results, writer_agent)
            logger.info("Report written, sending email...")

            await self.send_email(report,email_agent)
            logger.info("Email sent, research complete")

            return report

    async def plan_searches(self, query: str, planner_agent: Agent) -> WebSearchPlan:
        """Plan searches for the supplied query."""

        logger.info("Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        logger.info(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan,search_agent:Agent) -> list[str]:
        """Execute searches concurrently."""

        logger.info("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item,search_agent)) for item in search_plan.searches]
        results: list[str] = []

        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            logger.info(f"Searching... {num_completed}/{len(tasks)} completed")

        logger.info("Finished searching")
        return results

    async def search(self, item: WebSearchItem,search_agent:Agent) -> Optional[str]:
        """Perform a single search with the search agent."""

        payload = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(search_agent, payload)
            return str(result.final_output)
        except Exception as e:  # pragma: no cover - agent failures delegated for now
            logger.error(f"Search failed for {item.query}: {e}")
            return None

    async def write_report(self, query: str, search_results: list[str], writer_agent: Agent) -> ReportData:
        """Use the writer agent to produce a report."""

        logger.info("Thinking about report...")
        payload = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(writer_agent, payload)
        logger.info("Finished writing report")
        return result.final_output_as(ReportData)

    async def send_email(self, report: ReportData, email_agent: Agent) -> ReportData:
        """Delegate email composition to the email agent."""

        logger.info("Writing email...")
        await Runner.run(email_agent, report.markdown_report)
        logger.info("Email sent")
        return report


@router.post("/run", response_model=ReportData)
async def run_deep_research(payload: DeepResearchRequest) -> JSONResponse:
    """Run deep research and return the final report."""

    manager = ResearchManager()

    try:
        report = await manager.run(payload)
        return JSONResponse(content={"report": report})
    except Exception as e:
        logger.error(f"Deep research failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research process failed. Check logs for details."
        )

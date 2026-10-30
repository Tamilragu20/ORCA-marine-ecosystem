"""
ORCA Multi-Agent Orchestration & Workflow Engine
Executes the sequential, structured collaboration pipeline across all specialized agents.
"""

import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from ml.preprocessing import MarineDataValidator
from agents.water_quality_agent import WaterQualityAgent
from agents.biodiversity_agent import BiodiversityAgent
from agents.productivity_agent import OceanProductivityAgent
from agents.ml_agent import MLAgent
from agents.risk_agent import RiskAgent
from agents.reasoning_agent import ReasoningAgent
from agents.report_agent import ReportAgent


class PipelineStepLog(BaseModel):
    step_number: int
    agent_id: str
    display_name: str
    status: str
    severity: str
    execution_time_ms: float
    summary: str


class OrchestrationResult(BaseModel):
    session_id: str
    timestamp: str
    overall_status: str
    final_assessment: str
    final_confidence: float
    pipeline_steps: List[PipelineStepLog]
    agent_results: Dict[str, Any]
    input_environmental: Dict[str, Any]
    report_markdown: str
    total_execution_time_ms: float


class ORCAOrchestrator:
    """Coordinates the collaborative execution sequence of the ORCA multi-agent architecture."""

    def __init__(self):
        print("Initializing ORCA Multi-Agent Workflow Engine...")
        self.water_agent = WaterQualityAgent()
        self.bio_agent = BiodiversityAgent()
        self.prod_agent = OceanProductivityAgent()
        self.ml_agent = MLAgent()
        self.risk_agent = RiskAgent()
        self.reasoning_agent = ReasoningAgent()
        self.report_agent = ReportAgent()

    def run_pipeline(self, environmental_data: Dict[str, Any], session_id: str = "orca-session-001") -> OrchestrationResult:
        """
        Executes the linear collaborative workflow:
        Validation -> Water Quality -> Biodiversity -> Productivity -> ML -> Risk -> Coordinator -> Report
        """
        start_total = time.time()
        pipeline_logs: List[PipelineStepLog] = []
        agent_outputs: Dict[str, Any] = {}

        # 0. Data Validation Step
        is_valid, validation_issues = MarineDataValidator.validate_record(environmental_data)
        context = {
            "environmental": environmental_data,
            "validation": {"is_valid": is_valid, "issues": validation_issues},
            "agent_outputs": agent_outputs
        }

        # Sequential Agent Pipeline
        agents_sequence = [
            (1, self.water_agent),
            (2, self.bio_agent),
            (3, self.prod_agent),
            (4, self.ml_agent),
            (5, self.risk_agent),
            (6, self.reasoning_agent),
            (7, self.report_agent)
        ]

        for step_idx, agent in agents_sequence:
            res = agent.execute(context)
            agent_outputs[agent.agent_id] = res.model_dump()
            
            pipeline_logs.append(PipelineStepLog(
                step_number=step_idx,
                agent_id=agent.agent_id,
                display_name=agent.display_name,
                status=res.status,
                severity=res.severity,
                execution_time_ms=res.execution_time_ms,
                summary=res.summary
            ))

        coord_out = agent_outputs.get("coordinator_reasoning", {})
        final_assessment = coord_out.get("metrics", {}).get("final_assessment", "Assessment Complete")
        final_conf = coord_out.get("metrics", {}).get("confidence", 0.95)
        report_md = agent_outputs.get("report_generator", {}).get("metrics", {}).get("report_markdown", "")

        total_time_ms = round((time.time() - start_total) * 1000, 2)

        return OrchestrationResult(
            session_id=session_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            overall_status="SUCCESS",
            final_assessment=final_assessment,
            final_confidence=final_conf,
            pipeline_steps=pipeline_logs,
            agent_results=agent_outputs,
            input_environmental=environmental_data,
            report_markdown=report_md,
            total_execution_time_ms=total_time_ms
        )


# Singleton orchestrator
_orchestrator = None

def get_orchestrator() -> ORCAOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ORCAOrchestrator()
    return _orchestrator

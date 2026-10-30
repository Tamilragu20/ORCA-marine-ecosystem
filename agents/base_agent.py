"""
ORCA Base Agent Definition
Defines the standard structured communication protocol and base agent class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time


class AgentFinding(BaseModel):
    """Structured individual finding from an agent."""
    indicator: str
    observed_value: Any
    baseline_reference: Optional[str] = None
    status: str # "optimal", "warning", "critical", "informative"
    description: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class AgentResult(BaseModel):
    """Standardized payload exchanged across the ORCA multi-agent pipeline."""
    agent_id: str
    display_name: str
    status: str # "completed", "warning", "alert"
    severity: str # "low", "moderate", "high", "critical"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    summary: str
    findings: List[AgentFinding] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class BaseAgent(ABC):
    """Abstract base class for all specialized ORCA reasoning agents."""

    def __init__(self, agent_id: str, display_name: str):
        self.agent_id = agent_id
        self.display_name = display_name

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> AgentResult:
        """Executes agent-specific analytical logic over the incoming context."""
        pass

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Wraps execution with performance timing and error handling."""
        start = time.time()
        try:
            result = self.run(context)
        except Exception as e:
            result = AgentResult(
                agent_id=self.agent_id,
                display_name=self.display_name,
                status="alert",
                severity="high",
                confidence=0.5,
                summary=f"Agent encountered analytical error: {str(e)}",
                findings=[
                    AgentFinding(
                        indicator="execution_error",
                        observed_value=str(e),
                        status="critical",
                        description="Error during agent reasoning cycle.",
                        confidence=0.5
                    )
                ]
            )
        result.execution_time_ms = round((time.time() - start) * 1000, 2)
        return result

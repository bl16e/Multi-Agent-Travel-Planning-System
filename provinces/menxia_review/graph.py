from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from utils.agent_runtime import run_structured_synthesis, soul_path_for
from utils.schemas import MenxiaReviewPacketModel, ReviewVerdictModel, ZhongshuDraftPacketModel


class MenxiaState(TypedDict, total=False):
    request_id: str
    draft: dict[str, Any]
    user_request: dict[str, Any]
    parsed_draft: dict[str, Any]
    needs_human: bool
    human_question: str
    verdict_payload: dict[str, Any]


class MenxiaReviewAgent:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(MenxiaState)
        graph.add_node("ingest_draft", self.ingest_draft)
        graph.add_node("review_draft", self.review_draft)
        graph.add_node("human_gate", self.human_gate)
        graph.add_node("verdict", self.verdict)
        graph.set_entry_point("ingest_draft")
        graph.add_edge("ingest_draft", "review_draft")
        graph.add_conditional_edges("review_draft", self.route_after_review, {"human_gate": "human_gate", "verdict": "verdict"})
        graph.add_edge("human_gate", "verdict")
        graph.add_edge("verdict", END)
        return graph.compile()

    async def ingest_draft(self, state: MenxiaState) -> dict[str, Any]:
        packet = ZhongshuDraftPacketModel.model_validate(state["draft"])
        return {"parsed_draft": packet.model_dump(mode="json")}

    async def review_draft(self, state: MenxiaState) -> dict[str, Any]:
        profile = state.get("user_request", {}).get("profile", {})
        if profile.get("total_budget") is None:
            return {"needs_human": True, "human_question": "Please confirm the overall trip budget cap before approval."}
        return {"needs_human": False}

    def route_after_review(self, state: MenxiaState) -> str:
        return "human_gate" if state.get("needs_human") else "verdict"

    async def human_gate(self, state: MenxiaState) -> dict[str, Any]:
        response = interrupt({"type": "menxia_human_intervention", "request_id": state["request_id"], "question": state.get("human_question", "Menxia requires user confirmation.")})
        user_request = dict(state.get("user_request", {}))
        profile = dict(user_request.get("profile", {}))
        if isinstance(response, dict):
            profile.update(response.get("profile_updates", {}))
        user_request["profile"] = profile
        return {"user_request": user_request}

    async def verdict(self, state: MenxiaState) -> dict[str, Any]:
        packet = ZhongshuDraftPacketModel.model_validate(state["parsed_draft"])
        profile = state.get("user_request", {}).get("profile", {})

        gov_dict = packet.governance.model_dump() if hasattr(packet.governance, 'model_dump') else packet.governance
        revision_round = int(gov_dict.get("revision_round", 0))

        if profile.get("total_budget") is None:
            verdict = ReviewVerdictModel(verdict="HUMAN_INTERVENE", summary="预算信息缺失，需要用户确认。", blocking_issues=["总预算未设置"], human_questions=["请提供旅行总预算"], review_notes=[])
        else:
            verdict = await run_structured_synthesis(
                soul_path=self.soul_path,
                output_model=ReviewVerdictModel,
                user_prompt=(
                    "审核中书省提交的行程草案，返回结构化审核结果。\n"
                    "草案内容: {draft}\n"
                    "用户需求: {user_request}\n\n"
                    "审核标准：\n"
                    "- 是否包含具体景点名称（非泛指区域）\n"
                    "- 是否有真实预订链接\n"
                    "- 是否有交通细节和时长\n"
                    "- 是否有天气应急方案\n\n"
                    "重要：你只能返回 APPROVED（通过）或 REJECTED（拒绝）。\n"
                    "不要返回 HUMAN_INTERVENE，预订确认、餐厅预约等执行细节不需要用户介入。\n"
                    "如果草案质量不足，使用 REJECTED 并在 revision_requests 中说明需要修改的内容。"
                ),
                variables={
                    "draft": str(packet.model_dump(mode="json")),
                    "user_request": str(state.get("user_request", {})),
                },
                timeout_seconds=200.0,
            )
            # Guard: AI should not return HUMAN_INTERVENE; downgrade to REJECTED
            if verdict.verdict == "HUMAN_INTERVENE":
                verdict = ReviewVerdictModel(
                    verdict="REJECTED",
                    summary=verdict.summary,
                    blocking_issues=verdict.blocking_issues,
                    revision_requests=verdict.revision_requests + verdict.human_questions,
                    human_questions=[],
                    review_notes=verdict.review_notes + ["[auto-downgraded from HUMAN_INTERVENE]"],
                )

        packet_out = MenxiaReviewPacketModel.model_validate({
            "request_id": packet.request_id,
            "verdict": verdict.verdict,
            "summary": verdict.summary,
            "blocking_issues": verdict.blocking_issues,
            "revision_requests": verdict.revision_requests,
            "human_questions": verdict.human_questions,
            "approved_bureaus": verdict.approved_bureaus,
            "governance": {
                "reviewer": "MENXIA",
                "source_producer": "ZHONGSHU",
                "next_hop": "SHANGSHU" if verdict.verdict == "APPROVED" else "ZHONGSHU",
                "verdict_state": verdict.verdict,
                "veto_enabled": True,
                "rejection_round": revision_round + (1 if verdict.verdict == "REJECTED" else 0),
                "max_rejection_rounds": 2
            },
            "review_notes": verdict.review_notes,
        })
        return {"verdict_payload": packet_out.model_dump(mode="json")}

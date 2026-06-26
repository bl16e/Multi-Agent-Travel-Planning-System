from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import ThreeProvinceTravelSystem
from utils.schemas import PlanningRequest


def build_request(request_id: str, days_from_today: int) -> PlanningRequest:
    start = date.today() + timedelta(days=days_from_today)
    return PlanningRequest.model_validate(
        {
            "request_id": request_id,
            "user_message": (
                "\u4e3a\u5317\u4eac\u51fa\u53d1\u7684\u4e0a\u6d77\u77ed\u9014\u65c5\u884c"
                "\u751f\u6210\u53ef\u4ea4\u4ed8\u8ba1\u5212\uff0c\u5fc5\u987b\u4f7f\u7528"
                "\u5177\u4f53\u5730\u70b9\u3001\u4ea4\u901a\u8bf4\u660e\u548c\u53ef\u65e5"
                "\u5386\u5316\u65f6\u95f4\uff0c\u907f\u514d\u6a21\u677f\u5316\u6d3b\u52a8\u3002"
            ),
            "profile": {
                "origin_city": "\u5317\u4eac",
                "origin_airport_code": "PEK",
                "destination_preferences": ["\u4e0a\u6d77"],
                "destination_airport_code": "PVG",
                "start_date": start.isoformat(),
                "end_date": start.isoformat(),
                "adults": 1,
                "children": 0,
                "budget_level": "mid_range",
                "total_budget": 1800,
                "currency": "CNY",
                "interests": ["\u6587\u5316", "\u7f8e\u98df"],
                "constraints": ["\u907f\u514d\u6cdb\u5316\u5360\u4f4d\u6d3b\u52a8"],
                "pace": "structured",
            },
        }
    )


def compact_summary(data: dict[str, Any]) -> dict[str, Any]:
    review = data.get("review") or {}
    return {
        "status": data.get("status") or data.get("workflow_state"),
        "request_id": data.get("request_id"),
        "reason": data.get("reason"),
        "review_verdict": review.get("verdict"),
        "review_source": review.get("data_source"),
        "fallback_sources": data.get("fallback_sources"),
        "markdown_file": str(data.get("markdown_file")),
        "calendar_file": str(data.get("calendar_file")),
    }


async def run_probe(request_id: str, days_from_today: int, output: Path) -> None:
    planner = ThreeProvinceTravelSystem()
    result = await planner.plan_trip(build_request(request_id, days_from_today))
    data = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(compact_summary(data), ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", default="real_domestic_shanghai_probe")
    parser.add_argument("--days-from-today", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_domestic_shanghai_probe_result.json"))
    args = parser.parse_args()
    asyncio.run(run_probe(args.request_id, args.days_from_today, args.output))


if __name__ == "__main__":
    main()

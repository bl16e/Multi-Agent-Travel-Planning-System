"""交互式运行脚本 - 支持 Human-in-Loop"""
import asyncio
import json
from datetime import date
from pathlib import Path

from main import ThreeProvinceTravelSystem
from utils.schemas import PlanningRequest


def console_progress(msg: str) -> None:
    print(msg)


async def interactive_demo():
    print("=== 三省六部旅行规划系统 - 交互模式 ===\n")

    system = ThreeProvinceTravelSystem(
        artifact_dir=Path("artifacts"),
        progress_reporter=console_progress
    )

    # 创建请求
    request = PlanningRequest.model_validate({
        "request_id": "interactive_demo_001",
        "user_message": "计划东京5日游",
        "profile": {
            "origin_city": "Beijing",
            "destination_preferences": ["Tokyo"],
            "destination_airport_code": "HND",
            "start_date": date(2026, 4, 18),
            "end_date": date(2026, 4, 22),
            "adults": 2,
            "budget_level": "mid_range",
            "total_budget": 3000,
            "currency": "USD",
            "interests": ["food", "culture"],
            "constraints": [],
            "pace": "balanced"
        }
    })

    print(f"请求ID: {request.request_id}")
    print(f"目的地: {request.profile.destination_preferences}")
    print(f"日期: {request.profile.start_date} 至 {request.profile.end_date}")
    print(f"预算: {request.profile.total_budget} {request.profile.currency}\n")

    # 首次运行
    result = await system.plan_trip(request)

    # 处理 Human-in-Loop
    while result.get("status") == "HUMAN_INTERVENE":
        print(f"\n{'='*60}")
        print(f"🤔 需要您的输入:")
        print(f"问题: {result.get('question')}")
        print(f"{'='*60}\n")

        user_input = input("请输入您的回答 (或输入 'skip' 跳过): ").strip()

        if user_input.lower() == 'skip':
            print("已跳过，流程结束。")
            break

        # 恢复流程
        resume_payload = {"profile_updates": {}, "user_response": user_input}
        result = await system.resume_trip(request.request_id, resume_payload)

    # 输出最终结果
    print(f"\n{'='*60}")
    print(f"✅ 流程完成！状态: {result.get('status')}")
    print(f"{'='*60}\n")

    if result.get("status") == "DONE":
        final_pkg = result.get("final_package", {})
        print(f"目的地: {final_pkg.get('destination')}")
        print(f"Markdown文件: {final_pkg.get('markdown_file')}")
        print(f"日历文件: {final_pkg.get('calendar_file')}")
        print(f"预订链接数: {len(final_pkg.get('booking_links', []))}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(interactive_demo())

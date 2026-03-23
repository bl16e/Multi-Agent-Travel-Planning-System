from __future__ import annotations

from typing import Any


def format_package_to_markdown(package: dict[str, Any]) -> str:
    """将旅行方案转换为中文 Markdown 格式"""
    md = []

    # 翻译映射
    theme_map = {"Arrival and orientation": "抵达与适应", "Deep exploration": "深度探索"}
    mode_map = {"transit": "公共交通", "flight": "飞机", "train": "火车", "car": "汽车", "walk": "步行", "mixed": "混合交通"}

    itinerary = package.get("itinerary", {})
    destination = itinerary.get("destination", package.get("destination", "未知目的地"))
    md.append(f"# {destination} 旅行方案\n\n")
    md.append(f"**行程概览**: {itinerary.get('overview', '暂无')}\n\n")
    md.append(f"**旅行风格**: {itinerary.get('trip_style', '暂无')}\n\n")

    # 每日行程
    md.append("## 📅 每日行程\n\n")
    for day in itinerary.get("daily_plan", []):
        theme = day.get('theme', '')
        theme_cn = theme_map.get(theme, theme)
        md.append(f"### Day {day.get('day_index')} - {day.get('date')} ({theme_cn})\n\n")
        md.append(f"**城市**: {day.get('city', '')}\n\n")

        for act in day.get("activities", []):
            md.append(f"#### {act.get('start_time')} - {act.get('end_time')}: {act.get('title')}\n\n")
            md.append(f"- **地点**: {act.get('location_name')}\n")
            md.append(f"- **描述**: {act.get('description')}\n")
            cost = act.get('estimated_cost', 0)
            if cost > 0:
                md.append(f"- **预估费用**: ${cost}\n")
            if act.get('map_link'):
                md.append(f"- **地图**: [查看地图]({act.get('map_link')})\n")
            if act.get('booking_link'):
                md.append(f"- **预订**: [预订链接]({act.get('booking_link')})\n")

            transport = act.get('transport')
            if transport:
                mode = str(transport.get('mode', '')).replace('TransportMode.', '').lower()
                mode_cn = mode_map.get(mode, mode)
                md.append(f"- **交通**: {mode_cn} ({transport.get('duration_text')}) - ${transport.get('estimated_cost', 0)}\n")
            md.append("\n")

        if day.get('accommodation_note'):
            md.append(f"**住宿建议**: {day.get('accommodation_note')}\n\n")

    # 预算
    budget = package.get("budget")
    if budget:
        md.append("## 💰 预算明细\n\n")
        total = budget.get('total_estimated_cost', 0)
        currency = budget.get('currency', 'USD')
        md.append(f"**总预算**: ${total} {currency}\n\n")
        breakdown = budget.get("budget_breakdown", [])
        if isinstance(breakdown, list):
            for item in breakdown:
                if isinstance(item, dict) and item.get('estimated_cost', 0) > 0:
                    md.append(f"- **{item.get('category', '未知')}**: ${item.get('estimated_cost')}\n")
        md.append("\n")

    # 天气
    weather = package.get("weather")
    if weather:
        md.append("## 🌤️ 天气预报\n\n")
        for day in weather.get("forecast_days", []):
            temp = day.get('temp_range', '')
            md.append(f"- **{day.get('date')}**: {day.get('condition', '未知')} {temp}\n")
        md.append("\n")

    # 住宿
    accommodation = package.get("accommodation")
    if accommodation:
        md.append("## 🏨 住宿推荐\n\n")
        for hotel in accommodation.get("hotel_options", [])[:3]:
            md.append(f"### {hotel.get('name')}\n\n")
            price = hotel.get('price_per_night') or hotel.get('nightly_rate')
            if price:
                md.append(f"- **价格**: ${price}/晚\n")
            if hotel.get('booking_link'):
                md.append(f"- **预订**: [预订链接]({hotel.get('booking_link')})\n")
            md.append("\n")

    # 航班
    flight = package.get("flight_transport")
    if flight:
        md.append("## ✈️ 航班信息\n\n")
        for option in flight.get("flight_options", [])[:2]:
            route = option.get('route') or f"{option.get('departure_airport', '')} → {option.get('arrival_airport', '')}"
            md.append(f"- **{route}**: {option.get('departure_time')} - {option.get('arrival_time')}\n")
            if option.get('booking_link'):
                md.append(f"  [预订链接]({option.get('booking_link')})\n")
        md.append("\n")

    return "".join(md)

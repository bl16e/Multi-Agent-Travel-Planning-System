from __future__ import annotations

from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

ExecutionStatus = Literal["ok", "fallback", "error"]
DataSource = Literal["live", "structured_llm", "fallback_estimate", "unavailable"]
LinkConfidence = Literal["canonical", "provider_result", "search_fallback", "unavailable"]


class AgentRole(StrEnum):
    USER = "USER"
    SHANGSHU = "SHANGSHU"
    ZHONGSHU = "ZHONGSHU"
    MENXIA = "MENXIA"
    WEATHER = "WEATHER"
    BUDGET = "BUDGET"
    ACCOMMODATION = "ACCOMMODATION"
    FLIGHT_TRANSPORT = "FLIGHT_TRANSPORT"
    CALENDAR = "CALENDAR"


class WorkflowState(StrEnum):
    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    HUMAN_INTERVENE = "HUMAN_INTERVENE"
    EXECUTE = "EXECUTE"
    ASSEMBLE = "ASSEMBLE"
    DONE = "DONE"


class ConfirmationStatus(str, Enum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    OPTIONAL = "optional"


class BudgetLevel(str, Enum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    LUXURY = "luxury"


class TransportMode(str, Enum):
    FLIGHT = "flight"
    TRAIN = "train"
    CAR = "car"
    WALK = "walk"
    TRANSIT = "transit"
    MIXED = "mixed"


class TravelerProfile(BaseModel):
    origin_city: str = Field(..., description="Departure city.")
    origin_airport_code: str | None = None
    destination_preferences: list[str] = Field(default_factory=list)
    destination_airport_code: str | None = None
    start_date: date
    end_date: date
    adults: int = 1
    children: int = 0
    budget_level: BudgetLevel = BudgetLevel.MID_RANGE
    total_budget: float | None = None
    currency: str = "USD"
    interests: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    pace: str = "balanced"

    @model_validator(mode="after")
    def validate_date_order(self) -> "TravelerProfile":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class PlanningRequest(BaseModel):
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Stable request identifier using only letters, digits, underscore, dash, and dot.",
    )
    user_message: str
    profile: TravelerProfile


class ProgressEvent(BaseModel):
    stage: str
    message: str
    actor: str
    state: str
    timestamp: datetime


class DepartmentTaskModel(BaseModel):
    target: AgentRole
    task_type: str
    payload: dict[str, Any]
    reason: str


class DashboardLinkModel(BaseModel):
    request_id: str
    url: HttpUrl | str


class ActivityTransportModel(BaseModel):
    from_location: str
    to_location: str
    mode: TransportMode = TransportMode.TRANSIT
    duration_text: str
    booking_link: HttpUrl | str | None = None
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    estimated_cost: float | None = None
    notes: str | None = None


class ActivityModel(BaseModel):
    start_time: str = Field(..., description="HH:MM local time")
    end_time: str = Field(..., description="HH:MM local time")
    title: str
    location_name: str
    description: str
    map_link: HttpUrl | str | None = None
    canonical_url: HttpUrl | str | None = None
    search_url: HttpUrl | str | None = None
    link_confidence: LinkConfidence = "unavailable"
    provider: str | None = None
    provider_place_id: str | None = None
    estimated_cost: float | None = None
    booking_link: HttpUrl | str | None = None
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    transport: ActivityTransportModel | None = None


class DayPlanModel(BaseModel):
    day_index: int
    date: date
    city: str
    theme: str
    summary: str
    activities: list[ActivityModel]
    accommodation_note: str | None = None


class BureauTaskSpec(BaseModel):
    bureau: Literal["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]
    objective: str
    inputs_required: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"


class ItineraryDraftModel(BaseModel):
    destination: str
    overview: str
    trip_style: str
    daily_plan: list[DayPlanModel]
    planning_notes: list[str] = Field(default_factory=list)
    pending_confirmations: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class SerpApiQueryModel(BaseModel):
    tool: Literal["search_google_maps", "search_local_places"] = "search_google_maps"
    query: str = Field(..., min_length=1)
    location: str = ""
    reason: str = ""


class SerpApiQueryPlanModel(BaseModel):
    queries: list[SerpApiQueryModel] = Field(default_factory=list, min_length=1, max_length=4)
    strategy_notes: list[str] = Field(default_factory=list)


class SerpApiCandidateSelectionItemModel(BaseModel):
    candidate_id: str = Field(..., min_length=1)
    title: str = ""
    reason: str = ""
    confidence: float = 0.0


class SerpApiCandidateSelectionModel(BaseModel):
    selected_candidates: list[SerpApiCandidateSelectionItemModel] = Field(default_factory=list, min_length=1, max_length=8)
    selection_notes: list[str] = Field(default_factory=list)


class DraftGovernanceModel(BaseModel):
    producer: str = AgentRole.ZHONGSHU.value
    next_hop: str = AgentRole.MENXIA.value
    review_required: bool = True
    source_state: str = WorkflowState.DRAFT.value
    self_check_notes: list[str] = Field(default_factory=list)
    human_intervened: bool = False
    revision_round: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)
    revision_requests: list[str] = Field(default_factory=list)


class ZhongshuDraftPacketModel(BaseModel):
    request_id: str
    destination: str
    itinerary_draft: ItineraryDraftModel
    required_bureaus: list[Literal["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]]
    bureau_tasks: list[BureauTaskSpec]
    governance: DraftGovernanceModel | dict[str, Any]


class ReviewVerdictModel(BaseModel):
    verdict: Literal["APPROVED", "REJECTED", "HUMAN_INTERVENE"]
    summary: str
    blocking_issues: list[str] = Field(default_factory=list)
    revision_requests: list[str] = Field(default_factory=list)
    human_questions: list[str] = Field(default_factory=list)
    approved_bureaus: list[Literal["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    data_source: DataSource = "structured_llm"
    warnings: list[str] = Field(default_factory=list)


class ReviewGovernanceModel(BaseModel):
    reviewer: str = AgentRole.MENXIA.value
    source_producer: str = AgentRole.ZHONGSHU.value
    next_hop: str
    verdict_state: str
    veto_enabled: bool = True
    rejection_round: int = 0
    max_rejection_rounds: int = 0


class MenxiaReviewPacketModel(BaseModel):
    request_id: str
    verdict: Literal["APPROVED", "REJECTED", "HUMAN_INTERVENE"]
    summary: str
    blocking_issues: list[str] = Field(default_factory=list)
    revision_requests: list[str] = Field(default_factory=list)
    human_questions: list[str] = Field(default_factory=list)
    approved_bureaus: list[Literal["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]] = Field(default_factory=list)
    governance: ReviewGovernanceModel | dict[str, Any]
    review_notes: list[str] = Field(default_factory=list)
    data_source: DataSource = "structured_llm"
    warnings: list[str] = Field(default_factory=list)


class WeatherDayModel(BaseModel):
    date: date
    condition: str
    min_temp_c: float
    max_temp_c: float
    precipitation_probability: float
    activity_suitability: str
    clothing_advice: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_estimated: bool = False


class WeatherExecutionResult(BaseModel):
    bureau: Literal["WEATHER"] = "WEATHER"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    destination: str
    forecast_days: list[WeatherDayModel]
    packing_list: list[str]
    warnings: list[str] = Field(default_factory=list)
    summary: str
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-friendly simplified weather schemas (short field names avoid Qwen
# inventing Chinese keys when generating structured output).
# ---------------------------------------------------------------------------


class SimpleWeatherDay(BaseModel):
    """Single forecast day – short keys so the LLM rarely misses or renames them."""

    date: str = ""  # "YYYY-MM-DD"
    cond: str = "Weather unavailable"  # condition
    lo: float = 18.0  # min_temp_c
    hi: float = 26.0  # max_temp_c
    rain: float = 0.2  # precipitation_probability (0-1)
    suit: str = "Use flexible scheduling."  # activity_suitability
    wear: list[str] = Field(default_factory=list)  # clothing_advice
    alert: list[str] = Field(default_factory=list)  # warnings
    est: bool = True  # is_estimated

    @field_validator("wear", "alert", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v: Any) -> list[str]:
        """Qwen sometimes emits a single string instead of a list – wrap it."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        return []


class SimpleWeatherOutput(BaseModel):
    """Top-level weather result the LLM must return – flat, minimal keys."""

    dest: str = ""  # destination city name
    days: list[SimpleWeatherDay] = Field(default_factory=list)
    pack: list[str] = Field(default_factory=list)  # packing_list
    warn: list[str] = Field(default_factory=list)  # warnings
    note: str = ""  # summary

    @field_validator("pack", "warn", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v: Any) -> list[str]:
        """Qwen sometimes emits a single string instead of a list – wrap it."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        return []

    def to_weather_result(self, *, destination: str) -> WeatherExecutionResult:
        """Convert the simple LLM output to the full domain model."""
        converted_days: list[WeatherDayModel] = []
        for d in self.days:
            parsed_date: date
            try:
                parsed_date = date.fromisoformat(d.date) if d.date else date.today()
            except ValueError:
                parsed_date = date.today()
            converted_days.append(
                WeatherDayModel(
                    date=parsed_date,
                    condition=d.cond or "Weather available",
                    min_temp_c=float(d.lo),
                    max_temp_c=float(d.hi),
                    precipitation_probability=float(d.rain),
                    activity_suitability=d.suit or "Use flexible scheduling.",
                    clothing_advice=d.wear or ["Pack light layers."],
                    warnings=d.alert or [],
                    is_estimated=bool(d.est),
                )
            )
        return WeatherExecutionResult(
            status="ok",
            data_source="structured_llm",
            destination=destination or self.dest,
            forecast_days=converted_days,
            packing_list=self.pack or ["phone charger", "comfortable walking shoes", "weather-appropriate layers"],
            warnings=self.warn or [],
            summary=self.note or "Weather guidance generated from live research context.",
        )


class CalendarEventModel(BaseModel):
    title: str
    start_at: datetime
    end_at: datetime
    location: str
    description: str
    url: HttpUrl | str | None = None
    reminders_minutes: list[int] = Field(default_factory=lambda: [60, 1440])


class CalendarEventListModel(BaseModel):
    events: list[CalendarEventModel] = Field(default_factory=list)


class CalendarExecutionResult(BaseModel):
    bureau: Literal["CALENDAR"] = "CALENDAR"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    calendar_file: Path
    events_created: int
    calendar_name: str
    warnings: list[str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)


class BudgetLineItemModel(BaseModel):
    category: str
    item: str
    estimated_cost: float
    currency: str
    notes: str | None = None


class BudgetExecutionResult(BaseModel):
    bureau: Literal["BUDGET"] = "BUDGET"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    currency: str
    budget_breakdown: list[BudgetLineItemModel]
    total_estimated_cost: float
    warnings: list[str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)


class HotelOptionModel(BaseModel):
    name: str
    nightly_rate: float | None = None
    total_rate: float | None = None
    currency: str
    rating: float | None = None
    booking_link: HttpUrl | str | None = None
    canonical_url: HttpUrl | str | None = None
    search_url: HttpUrl | str | None = None
    link_confidence: LinkConfidence = "unavailable"
    provider: str | None = None
    provider_place_id: str | None = None
    address: str | None = None
    notes: str | None = None


class AccommodationExecutionResult(BaseModel):
    bureau: Literal["ACCOMMODATION"] = "ACCOMMODATION"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    destination: str
    hotel_options: list[HotelOptionModel] = Field(default_factory=list)
    booking_links: list[HttpUrl | str] = Field(default_factory=list)
    search_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)
    liubu_quality: dict[str, Any] = Field(default_factory=dict)


class FlightOptionModel(BaseModel):
    airline: str
    price: float
    currency: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    arrival_time: str
    duration_minutes: int | None = None
    booking_link: HttpUrl | str | None = None
    canonical_url: HttpUrl | str | None = None
    search_url: HttpUrl | str | None = None
    link_confidence: LinkConfidence = "unavailable"
    provider: str | None = None
    notes: str | None = None


class FlightTransportExecutionResult(BaseModel):
    bureau: Literal["FLIGHT_TRANSPORT"] = "FLIGHT_TRANSPORT"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    origin: str
    destination: str
    flight_options: list[FlightOptionModel] = Field(default_factory=list)
    transport_notes: list[str] = Field(default_factory=list)
    booking_links: list[HttpUrl | str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)
    liubu_quality: dict[str, Any] = Field(default_factory=dict)


class FinalTravelPackageModel(BaseModel):
    request_id: str
    destination: str
    workflow_state: str = WorkflowState.DONE.value
    markdown_file: Path | None = None
    calendar_file: Path | None = None
    dashboard_url: HttpUrl | str | None = None
    itinerary: ItineraryDraftModel
    review: MenxiaReviewPacketModel
    weather: WeatherExecutionResult | None = None
    budget: BudgetExecutionResult | None = None
    accommodation: AccommodationExecutionResult | None = None
    flight_transport: FlightTransportExecutionResult | None = None
    booking_links: list[HttpUrl | str] = Field(default_factory=list)
    packing_list: list[str] = Field(default_factory=list)
    progress_events: list[ProgressEvent] = Field(default_factory=list)
    generated_at: datetime


class OrchestratorAssembledOutputModel(BaseModel):
    request_id: str
    workflow_state: str
    dashboard_url: HttpUrl | str
    draft: ZhongshuDraftPacketModel | dict[str, Any]
    review: MenxiaReviewPacketModel | dict[str, Any]
    execution_results: dict[str, Any] = Field(default_factory=dict)
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    state_history: list[dict[str, Any]] = Field(default_factory=list)



// ---- Backend schema types (mirrors utils/schemas.py) ----

export interface TravelerProfile {
  origin_city: string;
  origin_airport_code?: string | null;
  destination_preferences: string[];
  destination_airport_code?: string | null;
  start_date: string;
  end_date: string;
  adults: number;
  children: number;
  budget_level: "budget" | "mid_range" | "luxury";
  total_budget: number | null;
  currency: string;
  interests: string[];
  constraints: string[];
  pace: string;
}

export interface PlanningRequest {
  request_id: string;
  user_message: string;
  profile: TravelerProfile;
}

export interface ActivityTransport {
  from_location: string;
  to_location: string;
  mode: string;
  duration_text: string;
  booking_link?: string | null;
  estimated_cost?: number | null;
}

export interface Activity {
  start_time: string;
  end_time: string;
  title: string;
  location_name: string;
  description: string;
  map_link?: string | null;
  estimated_cost?: number | null;
  booking_link?: string | null;
  transport?: ActivityTransport | null;
}

export interface DayPlan {
  day_index: number;
  date: string;
  city: string;
  theme: string;
  summary: string;
  activities: Activity[];
  accommodation_note?: string | null;
}

export interface ItineraryDraft {
  destination: string;
  overview: string;
  trip_style: string;
  daily_plan: DayPlan[];
}

export interface BudgetLineItem {
  category: string;
  item: string;
  estimated_cost: number;
  currency: string;
  notes?: string | null;
}

export interface BudgetResult {
  bureau: "BUDGET";
  currency: string;
  budget_breakdown: BudgetLineItem[];
  total_estimated_cost: number;
  warnings: string[];
}

export interface WeatherDay {
  date: string;
  condition: string;
  min_temp_c: number;
  max_temp_c: number;
  precipitation_probability: number;
  activity_suitability: string;
  temp_range?: string;
}

export interface WeatherResult {
  bureau: "WEATHER";
  destination: string;
  forecast_days: WeatherDay[];
  packing_list: string[];
  summary: string;
}

export interface HotelOption {
  name: string;
  nightly_rate?: number | null;
  price_per_night?: number | null;
  total_rate?: number | null;
  currency: string;
  rating?: number | null;
  booking_link?: string | null;
  address?: string | null;
}

export interface AccommodationResult {
  bureau: "ACCOMMODATION";
  destination: string;
  hotel_options: HotelOption[];
  booking_links: string[];
}

export interface FlightOption {
  airline: string;
  price: number;
  currency: string;
  departure_airport: string;
  arrival_airport: string;
  departure_time: string;
  arrival_time: string;
  duration_minutes?: number | null;
  booking_link?: string | null;
  route?: string;
}

export interface FlightTransportResult {
  bureau: "FLIGHT_TRANSPORT";
  origin: string;
  destination: string;
  flight_options: FlightOption[];
  transport_notes: string[];
  booking_links: string[];
}

export interface ReviewPacket {
  summary: string;
  blocking_issues: string[];
  revision_requests: string[];
  human_questions: string[];
}

export interface FinalTravelPackage {
  request_id: string;
  destination: string;
  workflow_state: string;
  itinerary: ItineraryDraft;
  review: ReviewPacket;
  weather?: WeatherResult | null;
  budget?: BudgetResult | null;
  accommodation?: AccommodationResult | null;
  flight_transport?: FlightTransportResult | null;
  booking_links: string[];
  packing_list: string[];
  generated_at: string;
}

export interface HumanInterveneResult {
  status: "HUMAN_INTERVENE";
  request_id: string;
  question: string;
  dashboard_url?: string | null;
}

export interface RejectedResult {
  status: "REJECTED";
  request_id?: string;
  review?: ReviewPacket;
  blocking_issues?: string[];
  reason?: string;
  error?: string;
}

export type StreamResult = FinalTravelPackage | HumanInterveneResult | RejectedResult;

export type NodeState = "active" | "done" | "error";

export type SSEEventType = "progress" | "result" | "error" | "done";

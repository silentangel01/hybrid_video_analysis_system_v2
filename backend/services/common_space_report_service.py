"""Generate and persist common-space reports from persisted events."""

from collections import Counter
from datetime import datetime
import time
from typing import Any, Dict, List, Optional


class CommonSpaceReportService:
    def __init__(self, mongo_client):
        self.mongo = mongo_client
        self.collection = getattr(mongo_client, "collection", None)
        db = getattr(mongo_client, "db", None)
        self.reports_collection = db["common_space_reports"] if db is not None else None
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        if self.reports_collection is None:
            return

        self.reports_collection.create_index([("generated_at_ts", -1)])
        self.reports_collection.create_index([("report_key.stream_id", 1)])
        self.reports_collection.create_index([("report_key.url", 1)])
        self.reports_collection.create_index([("report_kind", 1)])

    def build_report(self, stream: Dict[str, Any], start_time: float, end_time: float) -> Dict[str, Any]:
        if start_time > end_time:
            start_time, end_time = end_time, start_time

        stream_id = str(stream.get("stream_id") or "").strip()
        camera_id = str(stream.get("camera_id") or "").strip()
        stream_url = str(stream.get("url") or "").strip()

        query: Dict[str, Any] = {
            "event_type": "common_space_utilization",
            "timestamp": {"$gte": start_time, "$lte": end_time},
        }
        clauses: List[Dict[str, Any]] = []
        if stream_id:
            clauses.append({"metadata.stream_id": stream_id})
        if stream_url:
            clauses.append({"metadata.stream_url": stream_url})
        if camera_id:
            clauses.append({"camera_id": camera_id})

        events: List[Dict[str, Any]] = []
        if self.collection is not None and clauses:
            query["$or"] = clauses
            events = list(self.collection.find(query).sort("timestamp", 1))

        return self._compose_report(stream, events, start_time, end_time)

    def save_generated_report(
        self,
        report: Dict[str, Any],
        llm_summary: Optional[str] = None,
        llm_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if self.reports_collection is None:
            return None

        generated_at_ts = time.time()
        report_kind = "llm" if llm_summary else "rule"
        summary_source = (
            str(llm_meta.get("model") or llm_meta.get("provider") or "llm")
            if llm_meta else
            str(report.get("generated_by") or "rule_based_v1")
        )

        record = {
            "report_type": "common_space",
            "report_kind": report_kind,
            "generated_at": datetime.utcnow(),
            "generated_at_ts": generated_at_ts,
            "summary_source": summary_source,
            "display_summary": llm_summary or str(report.get("narrative") or ""),
            "report_key": report.get("report_key") or {},
            "window": report.get("window") or {},
            "stats": report.get("stats") or {},
            "highlights": report.get("highlights") or {},
            "sample_events": report.get("sample_events") or [],
            "narrative": report.get("narrative"),
            "generated_by": report.get("generated_by"),
            "llm_summary": llm_summary,
            "llm_meta": llm_meta,
            "report": report,
        }

        result = self.reports_collection.insert_one(record)
        return str(result.inserted_id)

    def list_saved_reports(self, limit: int = 5) -> List[Dict[str, Any]]:
        if self.reports_collection is None:
            return []

        cursor = self.reports_collection.find(
            {"report_type": "common_space"}
        ).sort("generated_at_ts", -1).limit(max(1, limit))
        return list(cursor)

    def _compose_report(
        self,
        stream: Dict[str, Any],
        events: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        occupancy_counter: Counter = Counter()
        activity_counter: Counter = Counter()
        people_counts: List[int] = []
        safety_count = 0
        summaries: List[str] = []
        reasons: List[str] = []

        for event in events:
            summary = event.get("analysis_summary")
            if not isinstance(summary, dict):
                summary = {}

            occupancy = str(summary.get("space_occupancy") or "unknown").strip().lower()
            occupancy_counter[occupancy] += 1

            people = summary.get("estimated_people_count")
            if isinstance(people, (int, float)):
                people_counts.append(max(0, int(people)))

            activities = summary.get("activity_types")
            if isinstance(activities, list):
                for activity in activities:
                    text = str(activity or "").strip()
                    if text and text != "unknown":
                        activity_counter[text] += 1

            if summary.get("safety_concerns") is True:
                safety_count += 1

            scene_summary = str(summary.get("scene_summary") or "").strip()
            if scene_summary:
                summaries.append(scene_summary)

            occupancy_reason = str(summary.get("occupancy_reason") or "").strip()
            if occupancy_reason:
                reasons.append(occupancy_reason)

        dominant_occupancy = occupancy_counter.most_common(1)[0][0] if occupancy_counter else "unknown"
        avg_people = round(sum(people_counts) / len(people_counts), 1) if people_counts else 0
        peak_people = max(people_counts) if people_counts else 0
        top_activities = [
            {"name": name, "count": count}
            for name, count in activity_counter.most_common(5)
        ]
        occupancy_distribution = {
            "low": occupancy_counter.get("low", 0),
            "medium": occupancy_counter.get("medium", 0),
            "high": occupancy_counter.get("high", 0),
            "unknown": occupancy_counter.get("unknown", 0),
        }

        report = {
            "generated_by": "rule_based_v1",
            "report_key": {
                "stream_id": stream.get("stream_id"),
                "camera_id": stream.get("camera_id"),
                "url": stream.get("url"),
                "location": stream.get("location"),
                "lat_lng": stream.get("lat_lng"),
            },
            "window": {
                "start_time": start_time,
                "end_time": end_time,
                "event_count": len(events),
                "first_event_at": events[0].get("timestamp") if events else None,
                "last_event_at": events[-1].get("timestamp") if events else None,
            },
            "stats": {
                "dominant_occupancy": dominant_occupancy,
                "avg_people_count": avg_people,
                "peak_people_count": peak_people,
                "safety_event_count": safety_count,
                "occupancy_distribution": occupancy_distribution,
                "top_activities": top_activities,
            },
            "highlights": {
                "scene_summaries": self._dedupe(summaries, limit=3),
                "occupancy_reasons": self._dedupe(reasons, limit=3),
            },
            "sample_events": [
                {
                    "timestamp": event.get("timestamp"),
                    "image_url": event.get("image_url"),
                    "description": event.get("description"),
                    "analysis_summary": event.get("analysis_summary") or {},
                }
                for event in events[-5:]
            ],
        }
        report["narrative"] = self._build_narrative(report)
        return report

    def _build_narrative(self, report: Dict[str, Any]) -> str:
        event_count = report["window"]["event_count"]
        if event_count == 0:
            return "No common-space events were recorded for this stream in the selected time window."

        stats = report["stats"]
        parts = [
            f"This stream recorded {event_count} common-space events in the selected time window.",
            f"Overall utilization was predominantly {stats['dominant_occupancy']}.",
            f"Average visible people count was {stats['avg_people_count']}, peaking at {stats['peak_people_count']}.",
        ]

        top_activities = stats["top_activities"]
        if top_activities:
            names = ", ".join(item["name"] for item in top_activities[:3])
            parts.append(f"The most frequent observed activities were {names}.")

        if stats["safety_event_count"] > 0:
            parts.append(f"{stats['safety_event_count']} events included visible safety concerns.")
        else:
            parts.append("No visible safety concerns were flagged in the selected events.")

        scene_summaries = report["highlights"]["scene_summaries"]
        if scene_summaries:
            parts.append(f"Representative observation: {scene_summaries[0]}")

        occupancy_reasons = report["highlights"]["occupancy_reasons"]
        if occupancy_reasons:
            parts.append(f"Occupancy rationale: {occupancy_reasons[0]}")

        return " ".join(parts)

    @staticmethod
    def _dedupe(items: List[str], limit: int) -> List[str]:
        unique: List[str] = []
        for item in items:
            if item and item not in unique:
                unique.append(item)
            if len(unique) >= limit:
                break
        return unique

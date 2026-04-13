"""
Forex economic calendar — scrapes Forex Factory for upcoming events.

Provides:
  ForexCalendar.get_upcoming(pair, hours_ahead) → List[CalendarEvent]

Events are cached in memory and refreshed every 4 hours.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Currency extraction from pair string
_PAIR_CURRENCIES: Dict[str, tuple] = {
    "GBP/JPY": ("GBP", "JPY"),
    "EUR/JPY": ("EUR", "JPY"),
    "GBP/USD": ("GBP", "USD"),
    "USD/JPY": ("USD", "JPY"),
    "EUR/USD": ("EUR", "USD"),
}


@dataclass
class CalendarEvent:
    """A single economic calendar event."""
    time: str                   # ISO 8601 or human-readable time string
    currency: str               # e.g. "USD", "GBP"
    impact: str                 # "high", "medium", "low"
    event: str                  # e.g. "Non-Farm Payrolls"
    forecast: str               # e.g. "180K"
    previous: str               # e.g. "175K"
    actual: str = ""            # filled after release

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "currency": self.currency,
            "impact": self.impact,
            "event": self.event,
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
        }


class ForexCalendar:
    """
    Forex Factory economic calendar with in-memory caching.

    Usage::

        cal = ForexCalendar()
        events = cal.get_upcoming("GBP/JPY", hours_ahead=4)
        has_high = any(e.impact == "high" for e in events)
    """

    def __init__(self, refresh_hours: float = 4.0) -> None:
        self._cache: List[CalendarEvent] = []
        self._last_fetch: float = 0.0
        self._refresh_seconds = refresh_hours * 3600

    def get_upcoming(
        self,
        pair: str = "GBP/JPY",
        hours_ahead: int = 4,
    ) -> List[CalendarEvent]:
        """Return upcoming events for currencies in *pair* within *hours_ahead*."""
        self._maybe_refresh()

        # Determine relevant currencies
        currencies = set()
        if pair in _PAIR_CURRENCIES:
            currencies = set(_PAIR_CURRENCIES[pair])
        else:
            parts = pair.replace("_", "/").split("/")
            currencies = set(p.upper() for p in parts if len(p) == 3)

        if not currencies:
            return []

        # Filter events by currency
        relevant = [e for e in self._cache if e.currency.upper() in currencies]
        return relevant

    def get_all_cached(self) -> List[CalendarEvent]:
        """Return all cached events (for the current week)."""
        self._maybe_refresh()
        return list(self._cache)

    def has_high_impact(self, pair: str, hours_ahead: int = 1) -> bool:
        """Check if there's a high-impact event coming up."""
        events = self.get_upcoming(pair, hours_ahead)
        return any(e.impact == "high" for e in events)

    def _maybe_refresh(self) -> None:
        """Refresh cache if stale."""
        now = time.time()
        if now - self._last_fetch < self._refresh_seconds and self._cache:
            return
        self._fetch()

    def _fetch(self) -> None:
        """Fetch this week's calendar from Forex Factory."""
        try:
            events = self._fetch_forex_factory()
            self._cache = events
            self._last_fetch = time.time()
            logger.info("Calendar refreshed: %d events", len(events))
        except Exception as e:
            logger.warning("Calendar fetch failed: %s — using stale cache (%d events)",
                           e, len(self._cache))

    def _fetch_forex_factory(self) -> List[CalendarEvent]:
        """Scrape Forex Factory calendar page for this week's events."""
        events: List[CalendarEvent] = []
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "WaveTrader/1.0",
            })
            resp.raise_for_status()
            data = resp.json()

            for item in data:
                impact_raw = item.get("impact", "").lower().strip()
                if impact_raw in ("holiday",):
                    impact = "low"
                elif "high" in impact_raw:
                    impact = "high"
                elif "medium" in impact_raw:
                    impact = "medium"
                else:
                    impact = "low"

                events.append(CalendarEvent(
                    time=item.get("date", ""),
                    currency=item.get("country", "").upper(),
                    impact=impact,
                    event=item.get("title", ""),
                    forecast=str(item.get("forecast", "")),
                    previous=str(item.get("previous", "")),
                    actual=str(item.get("actual", "")),
                ))

        except Exception as e:
            logger.warning("Forex Factory JSON API failed: %s — trying HTML fallback", e)
            events = self._fetch_ff_html_fallback()

        return events

    def _fetch_ff_html_fallback(self) -> List[CalendarEvent]:
        """Fallback: parse Forex Factory HTML calendar."""
        try:
            from bs4 import BeautifulSoup

            url = "https://www.forexfactory.com/calendar"
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            })
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            events: List[CalendarEvent] = []

            rows = soup.select("tr.calendar__row")
            current_date = ""
            for row in rows:
                # Date cell
                date_cell = row.select_one("td.calendar__date span")
                if date_cell and date_cell.text.strip():
                    current_date = date_cell.text.strip()

                # Time
                time_cell = row.select_one("td.calendar__time")
                time_str = time_cell.text.strip() if time_cell else ""

                # Currency
                curr_cell = row.select_one("td.calendar__currency")
                currency = curr_cell.text.strip().upper() if curr_cell else ""

                # Impact
                impact_cell = row.select_one("td.calendar__impact span")
                impact = "low"
                if impact_cell:
                    cls = " ".join(impact_cell.get("class", []))
                    if "high" in cls or "red" in cls:
                        impact = "high"
                    elif "medium" in cls or "ora" in cls:
                        impact = "medium"

                # Event name
                event_cell = row.select_one("td.calendar__event span")
                event_name = event_cell.text.strip() if event_cell else ""

                # Forecast / Previous
                forecast_cell = row.select_one("td.calendar__forecast span")
                forecast = forecast_cell.text.strip() if forecast_cell else ""

                previous_cell = row.select_one("td.calendar__previous span")
                previous = previous_cell.text.strip() if previous_cell else ""

                actual_cell = row.select_one("td.calendar__actual span")
                actual = actual_cell.text.strip() if actual_cell else ""

                if currency and event_name:
                    events.append(CalendarEvent(
                        time=f"{current_date} {time_str}".strip(),
                        currency=currency,
                        impact=impact,
                        event=event_name,
                        forecast=forecast,
                        previous=previous,
                        actual=actual,
                    ))

            return events
        except ImportError:
            logger.warning("beautifulsoup4 not installed — HTML fallback unavailable")
            return []
        except Exception as e:
            logger.warning("HTML fallback failed: %s", e)
            return []


# ── Module-level singleton ────────────────────────────────────────────────────
_calendar: Optional[ForexCalendar] = None


def get_calendar() -> ForexCalendar:
    """Return the module-level ForexCalendar singleton."""
    global _calendar
    if _calendar is None:
        _calendar = ForexCalendar()
    return _calendar

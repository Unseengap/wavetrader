"""
LLM Decision Logger — JSONL logging for arbiter decisions.

Each decision is logged with full context.  When the trade closes,
the outcome is backfilled via ``log_outcome()``.  The accumulated
JSONL file becomes the fine-tuning dataset for a custom Gemini model.

File: ``logs/llm_decisions.jsonl``
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMDecisionLog:
    """
    Append-only JSONL logger for LLM arbiter decisions.

    Usage::

        log = LLMDecisionLog()
        log.log_decision(decision, context_snapshot)
        # ... later, when trade closes ...
        log.log_outcome(decision_id, {"pnl": 42.0, "result": "win"})
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "llm_decisions.jsonl"
        self._lock = threading.Lock()
        # In-memory index: decision_id → file line number (for outcome backfill)
        self._index: Dict[str, int] = {}
        self._line_count = 0
        self._load_index()

    def _load_index(self) -> None:
        """Scan existing log to build decision_id → line index."""
        if not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r") as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        did = entry.get("decision_id", "")
                        if did:
                            self._index[did] = i
                    except json.JSONDecodeError:
                        pass
                    self._line_count = i + 1
        except Exception as e:
            logger.warning("Failed to load decision log index: %s", e)

    def log_decision(self, decision_dict: dict, context_snapshot: Optional[dict] = None) -> None:
        """Append a decision entry to the JSONL log."""
        entry = {
            "decision_id": decision_dict.get("decision_id", ""),
            "timestamp": decision_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "model_id": decision_dict.get("model_id", ""),
            "pair": decision_dict.get("pair", ""),
            "original_signal": decision_dict.get("original_signal", ""),
            "original_confidence": decision_dict.get("original_confidence", 0),
            "action": decision_dict.get("action", ""),
            "reasoning": decision_dict.get("reasoning", ""),
            "confidence_adjustment": decision_dict.get("confidence_adjustment", 0),
            "modified_signal": decision_dict.get("modified_signal"),
            "modified_sl_pips": decision_dict.get("modified_sl_pips"),
            "modified_tp_pips": decision_dict.get("modified_tp_pips"),
            "risk_notes": decision_dict.get("risk_notes", ""),
            "model_used": decision_dict.get("model_used", ""),
            "latency_ms": decision_dict.get("latency_ms", 0),
            "entry_price": decision_dict.get("entry_price", 0),
            "trade_placed": decision_dict.get("trade_placed", False),
            "context": context_snapshot or {},
            "outcome": None,  # Filled later via log_outcome
        }

        with self._lock:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                self._index[entry["decision_id"]] = self._line_count
                self._line_count += 1
            except Exception as e:
                logger.error("Failed to log decision: %s", e)

    def log_outcome(self, decision_id: str, outcome: dict) -> None:
        """
        Backfill the outcome for a previously logged decision.

        outcome example: {"pnl": 42.0, "pips": 15.3, "result": "win",
                          "exit_reason": "TP hit", "duration_bars": 8}
        """
        if decision_id not in self._index:
            logger.debug("Decision %s not found in log — skipping outcome", decision_id)
            return

        with self._lock:
            try:
                # Read all lines, update the target, rewrite
                lines = self._log_path.read_text().splitlines()
                line_idx = self._index[decision_id]
                if line_idx < len(lines):
                    entry = json.loads(lines[line_idx])
                    entry["outcome"] = outcome
                    lines[line_idx] = json.dumps(entry)
                    self._log_path.write_text("\n".join(lines) + "\n")
                    logger.debug("Outcome logged for decision %s: %s", decision_id, outcome)
            except Exception as e:
                logger.error("Failed to log outcome for %s: %s", decision_id, e)

    def get_recent(self, count: int = 50) -> List[dict]:
        """Return the most recent N decisions (newest first)."""
        entries = []
        if not self._log_path.exists():
            return entries
        try:
            with open(self._log_path, "r") as f:
                all_lines = f.readlines()
            for line in reversed(all_lines[-count:]):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.error("Failed to read decision log: %s", e)
        return entries

    def get_stats(self) -> dict:
        """Return summary statistics of all logged decisions."""
        stats = {
            "total_decisions": 0,
            "approvals": 0,
            "vetoes": 0,
            "overrides": 0,
            "with_outcome": 0,
            "veto_would_have_won": 0,
            "veto_would_have_lost": 0,
            "override_won": 0,
            "override_lost": 0,
            "avg_latency_ms": 0,
        }

        if not self._log_path.exists():
            return stats

        latencies = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    stats["total_decisions"] += 1
                    action = entry.get("action", "")
                    if action == "APPROVE":
                        stats["approvals"] += 1
                    elif action == "VETO":
                        stats["vetoes"] += 1
                    elif action == "OVERRIDE":
                        stats["overrides"] += 1

                    latencies.append(entry.get("latency_ms", 0))

                    outcome = entry.get("outcome")
                    if outcome:
                        stats["with_outcome"] += 1
                        pnl = outcome.get("pnl", 0)

                        if action == "VETO":
                            # Simulate: if we had taken the trade, would it have won?
                            sim = outcome.get("simulated_pnl", pnl)
                            if sim > 0:
                                stats["veto_would_have_won"] += 1
                            else:
                                stats["veto_would_have_lost"] += 1
                        elif action == "OVERRIDE":
                            if pnl > 0:
                                stats["override_won"] += 1
                            else:
                                stats["override_lost"] += 1

        except Exception as e:
            logger.error("Failed to compute stats: %s", e)

        if latencies:
            stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        return stats


# ── Module-level singleton ────────────────────────────────────────────────────
_decision_log: Optional[LLMDecisionLog] = None


def get_decision_log() -> LLMDecisionLog:
    """Return the module-level LLMDecisionLog singleton."""
    global _decision_log
    if _decision_log is None:
        _decision_log = LLMDecisionLog()
    return _decision_log

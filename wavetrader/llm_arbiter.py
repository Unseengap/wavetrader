"""
LLM Signal Arbiter — Gemini 2.5 meta-decision layer.

Sits between model inference and trade execution.  Receives the model's
signal plus rich context (chart bars, portfolio, economic calendar, trade
history) and validates / adjusts / vetoes the signal.

Three authority modes:
  advisory  — log reasoning only, signal always executes unchanged
  veto      — can block a trade (force HOLD), cannot create or flip signals
  override  — can veto, adjust confidence, modify SL/TP, or flip direction
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data containers
# ─────────────────────────────────────────────────────────────────────────────

class AuthorityMode(Enum):
    ADVISORY = "advisory"
    VETO = "veto"
    OVERRIDE = "override"


class ArbiterAction(Enum):
    APPROVE = "APPROVE"
    VETO = "VETO"
    OVERRIDE = "OVERRIDE"


@dataclass
class LLMArbiterConfig:
    """Configuration for the LLM signal arbiter."""
    enabled: bool = True
    authority_mode: str = "override"            # always override
    model: str = "gemini-2.5-flash"             # gemini-2.5-flash or gemini-2.5-pro
    escalation_model: str = "gemini-2.5-pro"    # used for high-impact events
    api_key_env: str = "GEMINI_API_KEY"
    timeout: int = 15                           # seconds
    temperature: float = 0.2                    # low for deterministic decisions
    escalate_on_high_impact: bool = True        # use Pro for high-impact calendar events
    max_retries: int = 3
    recent_bars_count: int = 30                 # how many OHLCV bars to include in prompt
    recent_trades_count: int = 10               # how many past trades to include


@dataclass
class ArbiterContext:
    """All context provided to the LLM for decision-making."""
    # Model signal
    signal: str                     # BUY / SELL / HOLD
    confidence: float
    alignment: float
    sl_pips: float
    tp_pips: float
    entry_price: float
    model_id: str
    pair: str
    timeframe: str

    # Chart data — last N bars as list of dicts
    recent_bars: List[Dict[str, Any]] = field(default_factory=list)

    # Portfolio state
    balance: float = 0.0
    unrealized_pnl: float = 0.0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # Trade history — last N trades
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Calendar events
    calendar_events: List[Dict[str, Any]] = field(default_factory=list)
    has_high_impact_event: bool = False

    # Session info
    current_session: str = ""       # Tokyo / London / New York / Off-hours


@dataclass
class ArbiterDecision:
    """The LLM's decision on whether to execute the signal."""
    decision_id: str = ""
    action: str = "APPROVE"         # APPROVE / VETO / OVERRIDE
    reasoning: str = ""
    confidence_adjustment: float = 0.0   # additive: -0.2 = reduce by 20%
    modified_signal: Optional[str] = None   # only in OVERRIDE mode
    modified_sl_pips: Optional[float] = None
    modified_tp_pips: Optional[float] = None
    risk_notes: str = ""            # calendar/portfolio risk warnings
    model_used: str = ""            # which Gemini model was used
    latency_ms: float = 0.0
    timestamp: str = ""

    # Simulation tracking
    original_signal: str = ""
    original_confidence: float = 0.0
    entry_price: float = 0.0
    trade_placed: bool = False      # whether the trade was actually executed
    simulated_outcome: Optional[Dict[str, Any]] = None  # filled later


# ─────────────────────────────────────────────────────────────────────────────
# LLM Arbiter
# ─────────────────────────────────────────────────────────────────────────────

class LLMArbiter:
    """
    Google Gemini-powered signal arbiter.

    Usage::

        arbiter = LLMArbiter(LLMArbiterConfig(enabled=True))
        ctx = ArbiterContext(signal="BUY", confidence=0.72, ...)
        decision = arbiter.evaluate(ctx)
    """

    def __init__(self, config: LLMArbiterConfig) -> None:
        self.config = config
        self._client = None
        self._initialized = False

    def _ensure_client(self) -> bool:
        """Lazy-init the Google GenAI client."""
        if self._initialized:
            return self._client is not None
        self._initialized = True
        try:
            from google import genai
            api_key = os.environ.get(self.config.api_key_env, "")
            if not api_key:
                logger.warning("LLM Arbiter: %s not set — arbiter disabled", self.config.api_key_env)
                return False
            self._client = genai.Client(api_key=api_key)
            logger.info("LLM Arbiter initialized with %s", self.config.model)
            return True
        except ImportError:
            logger.warning("google-genai not installed — arbiter disabled")
            return False
        except Exception as e:
            logger.error("LLM Arbiter init failed: %s", e)
            return False

    def evaluate(self, context: ArbiterContext) -> ArbiterDecision:
        """
        Evaluate a trade signal. Returns an ArbiterDecision.
        On failure, returns APPROVE (graceful degradation).
        """
        decision_id = str(uuid.uuid4())[:12]
        t0 = time.time()

        # Default: approve (graceful degradation)
        fallback = ArbiterDecision(
            decision_id=decision_id,
            action="APPROVE",
            reasoning="LLM arbiter unavailable — defaulting to APPROVE",
            model_used="none",
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_signal=context.signal,
            original_confidence=context.confidence,
            entry_price=context.entry_price,
        )

        if not self.config.enabled:
            fallback.reasoning = "LLM arbiter disabled"
            return fallback

        if not self._ensure_client():
            return fallback

        try:
            # Choose model (escalate for high-impact events)
            model_name = self.config.model
            if self.config.escalate_on_high_impact and context.has_high_impact_event:
                model_name = self.config.escalation_model
                logger.info("Escalating to %s due to high-impact calendar event", model_name)

            # Build prompt
            prompt = self._build_prompt(context)
            system_instruction = self._system_instruction()

            # Call Gemini
            response = self._call_gemini(prompt, system_instruction, model_name)

            # Parse response
            decision = self._parse_response(response, context)
            decision.decision_id = decision_id
            decision.model_used = model_name
            decision.latency_ms = (time.time() - t0) * 1000
            decision.timestamp = datetime.now(timezone.utc).isoformat()
            decision.original_signal = context.signal
            decision.original_confidence = context.confidence
            decision.entry_price = context.entry_price

            # Enforce authority mode constraints
            decision = self._enforce_authority(decision)

            logger.info(
                "LLM Arbiter [%s]: %s → %s (%.0fms) reason=%s",
                self.config.authority_mode, context.signal,
                decision.action, decision.latency_ms,
                decision.reasoning[:80],
            )
            return decision

        except Exception as e:
            logger.error("LLM Arbiter evaluation failed: %s", e)
            fallback.latency_ms = (time.time() - t0) * 1000
            fallback.reasoning = f"LLM call failed: {e}"
            return fallback

    def _system_instruction(self) -> str:
        return (
            "You are a forex trade signal arbiter for an automated trading platform. "
            "You receive signals from AI models and must decide whether to APPROVE, VETO, or OVERRIDE them.\n\n"
            "RULES:\n"
            "1. APPROVE — the signal looks valid given market context. No changes needed.\n"
            "2. VETO — block the trade. Reasons: upcoming high-impact news, overextended drawdown, "
            "   conflicting higher-timeframe trend, choppy/ranging market, or concentrated risk.\n"
            "3. OVERRIDE — modify the signal. Only when you have HIGH confidence the adjustment is better. "
            "   You can: flip direction, adjust SL/TP, or adjust confidence.\n\n"
            "GUIDELINES:\n"
            "- Be conservative with VETO and OVERRIDE. When in doubt, APPROVE.\n"
            "- ALWAYS veto within 30 minutes of high-impact news events for the relevant currencies.\n"
            "- Consider the model's confidence and alignment scores — high alignment means multiple "
            "  timeframes agree.\n"
            "- Check trade history: if recent trades are mostly losses, be more cautious.\n"
            "- Consider session: signals during off-hours (Sydney session) with wide spreads deserve scrutiny.\n\n"
            "Respond ONLY with valid JSON matching this schema:\n"
            '{"action": "APPROVE"|"VETO"|"OVERRIDE", "reasoning": "string (2-4 sentences)", '
            '"confidence_adjustment": float (-0.3 to 0.3), '
            '"modified_signal": null|"BUY"|"SELL"|"HOLD", '
            '"modified_sl_pips": null|float, "modified_tp_pips": null|float, '
            '"risk_notes": "string (calendar/portfolio warnings)"}'
        )

    def _build_prompt(self, ctx: ArbiterContext) -> str:
        """Build the evaluation prompt from context."""
        lines = [
            f"## Signal to Evaluate",
            f"Model: {ctx.model_id} | Pair: {ctx.pair} | Timeframe: {ctx.timeframe}",
            f"Signal: **{ctx.signal}** | Confidence: {ctx.confidence:.4f} | Alignment: {ctx.alignment:.4f}",
            f"Entry: {ctx.entry_price:.5f} | SL: {ctx.sl_pips:.1f} pips | TP: {ctx.tp_pips:.1f} pips",
            "",
            f"## Market Context",
            f"Session: {ctx.current_session}",
        ]

        # Recent bars (compact table)
        if ctx.recent_bars:
            lines.append("")
            lines.append(f"## Last {len(ctx.recent_bars)} Bars (newest last)")
            lines.append("Time | Open | High | Low | Close | Volume")
            lines.append("---|---|---|---|---|---")
            for bar in ctx.recent_bars[-self.config.recent_bars_count:]:
                t = bar.get("time", bar.get("date", ""))
                lines.append(
                    f"{t} | {bar.get('open', 0):.5f} | {bar.get('high', 0):.5f} | "
                    f"{bar.get('low', 0):.5f} | {bar.get('close', 0):.5f} | {bar.get('volume', 0)}"
                )

        # Portfolio
        lines.extend([
            "",
            "## Portfolio State",
            f"Balance: ${ctx.balance:,.2f} | Unrealised P&L: ${ctx.unrealized_pnl:,.2f}",
            f"Max Drawdown: {ctx.max_drawdown:.2%} | Win Rate: {ctx.win_rate:.1%} | Total Trades: {ctx.total_trades}",
        ])

        if ctx.open_positions:
            lines.append(f"Open Positions: {len(ctx.open_positions)}")
            for pos in ctx.open_positions:
                lines.append(f"  - {pos.get('direction', '?')} {pos.get('instrument', '?')} @ {pos.get('price', 0):.5f} P&L: ${pos.get('unrealized_pnl', 0):.2f}")

        # Trade history
        if ctx.recent_trades:
            lines.extend(["", f"## Last {len(ctx.recent_trades)} Trades"])
            for t in ctx.recent_trades[:self.config.recent_trades_count]:
                pnl = t.get("realized_pl", t.get("pnl", 0))
                direction = t.get("direction", "?")
                result = "WIN" if float(pnl) > 0 else "LOSS"
                lines.append(
                    f"  {direction} | {t.get('open_time', '?')} → {t.get('close_time', '?')} | "
                    f"P&L: ${float(pnl):+.2f} ({result})"
                )

        # Calendar events
        if ctx.calendar_events:
            lines.extend(["", "## Upcoming Economic Events"])
            for ev in ctx.calendar_events:
                lines.append(
                    f"  [{ev.get('impact', '?').upper()}] {ev.get('time', '?')} — "
                    f"{ev.get('currency', '?')}: {ev.get('event', '?')} "
                    f"(Prev: {ev.get('previous', '?')}, Forecast: {ev.get('forecast', '?')})"
                )
        else:
            lines.extend(["", "## Upcoming Economic Events", "  None within the next 4 hours."])

        lines.extend([
            "",
            "## Decision Required",
            "Based on ALL the above, what is your decision? Respond with JSON only.",
        ])
        return "\n".join(lines)

    def _call_gemini(self, prompt: str, system_instruction: str, model_name: str, allow_thinking: bool = False) -> str:
        """Call the Gemini API and return the response text."""
        from google.genai import types

        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": self.config.temperature,
            "max_output_tokens": 4096 if allow_thinking else 1024,
        }
        # Gemini 2.5 models require thinking mode; use a small budget
        # for fast signal evals, uncapped for deep inspections
        if not allow_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                # resp.text can be None with thinking models — extract from parts as fallback
                text = response.text
                if not text and response.candidates:
                    parts = getattr(response.candidates[0].content, "parts", None)
                    if parts:
                        text = "".join(
                            p.text for p in parts
                            if getattr(p, "text", None) and not getattr(p, "thought", False)
                        )
                return text or ""
            except Exception as e:
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    logger.warning("Gemini API attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    raise

    def _parse_response(self, raw: str, context: ArbiterContext) -> ArbiterDecision:
        """Parse the LLM's JSON response into an ArbiterDecision."""
        decision = ArbiterDecision()

        # Extract JSON from response (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            # Remove code fence
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        decision.action = data.get("action", "APPROVE").upper()
        if decision.action not in ("APPROVE", "VETO", "OVERRIDE"):
            decision.action = "APPROVE"

        decision.reasoning = data.get("reasoning", "No reasoning provided")
        decision.risk_notes = data.get("risk_notes", "")

        # Confidence adjustment (clamp to safe range)
        adj = data.get("confidence_adjustment", 0.0)
        try:
            adj = float(adj)
        except (TypeError, ValueError):
            adj = 0.0
        decision.confidence_adjustment = max(-0.3, min(0.3, adj))

        # Override fields
        mod_sig = data.get("modified_signal")
        if mod_sig and isinstance(mod_sig, str) and mod_sig.upper() in ("BUY", "SELL", "HOLD"):
            decision.modified_signal = mod_sig.upper()

        mod_sl = data.get("modified_sl_pips")
        if mod_sl is not None:
            try:
                decision.modified_sl_pips = float(mod_sl)
            except (TypeError, ValueError):
                pass

        mod_tp = data.get("modified_tp_pips")
        if mod_tp is not None:
            try:
                decision.modified_tp_pips = float(mod_tp)
            except (TypeError, ValueError):
                pass

        return decision

    def _enforce_authority(self, decision: ArbiterDecision) -> ArbiterDecision:
        """Enforce authority mode constraints on the decision."""
        # Always override mode — no restrictions on LLM decisions
        return decision

    def apply_decision(
        self,
        signal_dict: dict,
        decision: ArbiterDecision,
    ) -> dict:
        """
        Apply the arbiter's decision to the signal dict.
        Returns the (potentially modified) signal dict.
        """
        if decision.action == "VETO":
            signal_dict = dict(signal_dict)
            signal_dict["signal"] = "HOLD"
            signal_dict["_arbiter_vetoed"] = True
            return signal_dict

        if decision.action == "OVERRIDE":
            signal_dict = dict(signal_dict)
            if decision.modified_signal:
                signal_dict["signal"] = decision.modified_signal
            if decision.confidence_adjustment:
                signal_dict["confidence"] = max(
                    0.0, min(1.0,
                             signal_dict.get("confidence", 0) + decision.confidence_adjustment)
                )
            if decision.modified_sl_pips is not None:
                signal_dict["sl_pips"] = decision.modified_sl_pips
            if decision.modified_tp_pips is not None:
                signal_dict["tp_pips"] = decision.modified_tp_pips
            signal_dict["_arbiter_overridden"] = True
            return signal_dict

        # APPROVE — no changes
        return signal_dict

    # ─────────────────────────────────────────────────────────────────────
    # Manual inspection — user-triggered comprehensive market analysis
    # ─────────────────────────────────────────────────────────────────────

    def inspect(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a comprehensive on-demand market inspection.

        Unlike ``evaluate()`` which validates a single signal, this method
        takes a broad snapshot of ALL models, accounts, open trades, calendar,
        and recent price action and asks the LLM to:
          1. Analyse the current market
          2. Comment on each model's behaviour
          3. Optionally recommend a trade (executed through a model account)

        Returns a dict with the LLM's full analysis + optional trade action.
        """
        t0 = time.time()
        result: Dict[str, Any] = {
            "inspection_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis": "",
            "trade_action": None,    # None or {model_id, signal, sl_pips, tp_pips, reasoning}
            "risk_warnings": [],
            "model_used": "",
            "latency_ms": 0,
            "error": None,
        }

        if not self._ensure_client():
            result["error"] = "Gemini client unavailable"
            result["analysis"] = "LLM arbiter could not initialise — check GEMINI_API_KEY."
            return result

        try:
            prompt = self._build_inspection_prompt(context)
            system = self._inspection_system_instruction(context)

            # Always use Pro for inspections since user explicitly requested it
            model_name = self.config.escalation_model or self.config.model
            result["model_used"] = model_name

            raw = self._call_gemini(prompt, system, model_name, allow_thinking=True)
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)

            # Parse the response
            parsed = self._parse_inspection_response(raw)
            result["analysis"] = parsed.get("analysis", raw)
            result["risk_warnings"] = parsed.get("risk_warnings", [])
            result["trade_action"] = parsed.get("trade_action")

            logger.info(
                "LLM Inspection complete (%.0fms): trade_action=%s",
                result["latency_ms"],
                result["trade_action"]["signal"] if result.get("trade_action") else "NONE",
            )

        except Exception as e:
            logger.error("LLM Inspection failed: %s", e)
            result["error"] = str(e)
            result["analysis"] = f"Inspection failed: {e}"
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)

        return result

    def _inspection_system_instruction(self, context: Dict[str, Any]) -> str:
        models_info = context.get("models_info", {})
        model_descriptions = ""
        for mid, info in models_info.items():
            model_descriptions += f"\n- **{info.get('name', mid)}** ({mid}): {info.get('description', 'AI trading model')}"

        return (
            "You are a senior forex market analyst and trade advisor for the WaveTrader automated trading platform.\n"
            "The user has pressed the 'Inspect Market' button to get your comprehensive analysis.\n\n"
            "## Platform Models" + model_descriptions + "\n\n"
            "## Your Role\n"
            "1. Analyse the current market conditions based on recent price action, calendar events, and session.\n"
            "2. Comment on open trades across ALL models — are they well-positioned? Should any be closed?\n"
            "3. Review each model's recent signals and trade history. Are they performing well?\n"
            "4. If you see a trading opportunity that the models might be missing, recommend a trade.\n"
            "5. Flag any risk concerns (news events, drawdown, overexposure, session risks).\n\n"
            "## Trade Recommendations\n"
            "If you recommend a trade, it will be executed through one of the model's OANDA accounts.\n"
            "Choose the most appropriate model account (prefer the one with the best balance/margin).\n"
            "Only recommend trades with a clear setup — DO NOT force a trade if the market is unclear.\n\n"
            "## Response Format\n"
            "Respond with valid JSON matching this schema:\n"
            "```json\n"
            "{\n"
            '  "analysis": "string — 3-6 paragraph comprehensive market analysis",\n'
            '  "risk_warnings": ["string array — specific warnings"],\n'
            '  "trade_action": null | {\n'
            '    "model_id": "string — which model account to use",\n'
            '    "signal": "BUY"|"SELL",\n'
            '    "sl_pips": float,\n'
            '    "tp_pips": float,\n'
            '    "confidence": float (0-1),\n'
            '    "reasoning": "string — why this trade"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Set trade_action to null if no trade is recommended."
        )

    def _build_inspection_prompt(self, ctx: Dict[str, Any]) -> str:
        lines = [
            "# Market Inspection Request",
            f"Pair: **{ctx.get('pair', 'GBP/JPY')}** | Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Session: {ctx.get('current_session', 'Unknown')}",
            "",
        ]

        # Recent bars
        bars = ctx.get("recent_bars", [])
        if bars:
            lines.append(f"## Last {len(bars)} Bars (15min, newest last)")
            lines.append("Time | Open | High | Low | Close | Volume")
            lines.append("---|---|---|---|---|---")
            for bar in bars[-40:]:
                t = bar.get("time", bar.get("date", ""))
                lines.append(
                    f"{t} | {bar.get('open', 0):.5f} | {bar.get('high', 0):.5f} | "
                    f"{bar.get('low', 0):.5f} | {bar.get('close', 0):.5f} | {bar.get('volume', 0)}"
                )

        # Per-model state
        models = ctx.get("models", {})
        if models:
            lines.extend(["", "## Model Accounts & Positions"])
            for mid, m in models.items():
                lines.append(f"\n### {m.get('name', mid)} ({mid})")
                lines.append(f"Balance: ${m.get('balance', 0):,.2f} | NAV: ${m.get('nav', 0):,.2f} | Unrealised P&L: ${m.get('unrealized_pnl', 0):,.2f}")

                positions = m.get("open_positions", [])
                if positions:
                    lines.append(f"Open Positions ({len(positions)}):")
                    for pos in positions:
                        lines.append(
                            f"  - {pos.get('direction', '?')} {pos.get('instrument', '?')} "
                            f"@ {pos.get('price', 0):.3f} | P&L: ${pos.get('unrealized_pnl', 0):.2f} | "
                            f"SL: {pos.get('sl', 'N/A')} TP: {pos.get('tp', 'N/A')}"
                        )
                else:
                    lines.append("No open positions.")

                recent_trades = m.get("recent_trades", [])
                if recent_trades:
                    lines.append(f"Last {len(recent_trades)} trades:")
                    for t in recent_trades:
                        pnl = t.get("realized_pl", t.get("pnl", 0))
                        result = "WIN" if float(pnl) > 0 else "LOSS"
                        lines.append(
                            f"  {t.get('direction', '?')} | {t.get('open_time', t.get('openTime', '?'))} | "
                            f"P&L: ${float(pnl):+.2f} ({result})"
                        )

                recent_signals = m.get("recent_signals", [])
                if recent_signals:
                    lines.append(f"Recent signals:")
                    for s in recent_signals[-5:]:
                        lines.append(
                            f"  {s.get('signal', '?')} conf={s.get('confidence', 0):.3f} "
                            f"@ {s.get('price', 0):.3f} ({s.get('timestamp', '?')})"
                        )

        # Calendar
        events = ctx.get("calendar_events", [])
        if events:
            lines.extend(["", "## Upcoming Economic Events"])
            for ev in events:
                lines.append(
                    f"  [{ev.get('impact', '?').upper()}] {ev.get('time', '?')} — "
                    f"{ev.get('currency', '?')}: {ev.get('event', '?')} "
                    f"(Prev: {ev.get('previous', '?')}, Forecast: {ev.get('forecast', '?')})"
                )
        else:
            lines.extend(["", "## Upcoming Economic Events", "  None in the next 4 hours."])

        lines.extend([
            "",
            "## Analysis Required",
            "Provide your comprehensive market analysis. If you see a trade opportunity, include trade_action. Otherwise set it to null.",
        ])
        return "\n".join(lines)

    def _parse_inspection_response(self, raw: str) -> Dict[str, Any]:
        """Parse the LLM inspection response."""
        text = raw.strip()
        if text.startswith("```"):
            inner_lines = text.split("\n")
            text = "\n".join(inner_lines[1:-1] if inner_lines[-1].strip() == "```" else inner_lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return {"analysis": raw, "risk_warnings": [], "trade_action": None}
            else:
                return {"analysis": raw, "risk_warnings": [], "trade_action": None}

        result: Dict[str, Any] = {
            "analysis": data.get("analysis", ""),
            "risk_warnings": data.get("risk_warnings", []),
            "trade_action": None,
        }

        # Validate trade action if present
        ta = data.get("trade_action")
        if ta and isinstance(ta, dict):
            signal = (ta.get("signal") or "").upper()
            if signal in ("BUY", "SELL") and ta.get("model_id"):
                result["trade_action"] = {
                    "model_id": ta["model_id"],
                    "signal": signal,
                    "sl_pips": float(ta.get("sl_pips", 20)),
                    "tp_pips": float(ta.get("tp_pips", 40)),
                    "confidence": max(0.0, min(1.0, float(ta.get("confidence", 0.7)))),
                    "reasoning": ta.get("reasoning", ""),
                }

        return result

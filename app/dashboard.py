"""HealthTwin clinic screen.

The morning job is a ranked list. Open a person only when someone needs a look.
Run from the repository root:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitInvalidLayoutContextError

import importlib

from src import explain as explain_mod

importlib.reload(explain_mod)
from src.explain import explain_row
from src.train import train
from src.view import (
    artifacts_ready,
    clinic_times,
    forecast_path,
    history_window,
    load_bundle,
    snapshot_ward,
    view_model,
    what_if,
)

st.set_page_config(
    page_title="HealthTwin",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 0.35rem; padding-bottom: 0.4rem; max-width: 1400px;}
    header[data-testid="stHeader"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    .gt-kicker {color: #0F6E6E; font-size: 0.78rem; letter-spacing: 0.14em;
                text-transform: uppercase; font-weight: 700; margin-bottom: 0.15rem;}
    .gt-title {font-size: 1.45rem; font-weight: 750; color: #102124; margin: 0; white-space: nowrap;}
    .gt-when {color: #2c4246; font-size: 0.98rem; font-weight: 600; margin-left: 0.35rem; white-space: nowrap;}
    .gt-card {background: white; border: 1px solid #e2e8e8; border-radius: 12px;
              padding: 0.55rem 0.8rem; height: 100%;}
    .gt-card .label {color: #2c4246; font-size: 0.78rem; text-transform: uppercase;
                      letter-spacing: 0.05em; font-weight: 700;}
    .gt-card .value {font-size: 1.45rem; font-weight: 720; color: #102124; line-height: 1.15;}
    .gt-card .old {color: #3d5156; font-size: 0.92rem; text-decoration: line-through;
                   margin-top: 0.05rem;}
    .gt-card .hint {color: #2c4246; font-size: 0.88rem; font-weight: 500;}
    .gt-card .gt-tag {display: inline-block; margin-left: 0.25rem; padding: 0.05rem 0.4rem;
                      border-radius: 999px; font-size: 0.68rem; letter-spacing: 0.04em;
                      text-transform: uppercase; font-weight: 700; vertical-align: 0.1rem;}
    .gt-card.danger {border-color: #f0c3bd; background: #fff6f5;}
    .gt-card.danger .value {color: #9f2a1f;}
    .gt-card.danger .gt-tag {background: #9f2a1f; color: white;}
    .gt-card.warn {border-color: #f3d7a8; background: #fffaf2;}
    .gt-card.warn .value {color: #8a5a08;}
    .gt-card.warn .gt-tag {background: #8a5a08; color: white;}
    .gt-card.ok {border-color: #c6e6d4; background: #f4fbf7;}
    .gt-card.ok .value {color: #0d6b45;}
    .gt-card.ok .gt-tag {background: #0d6b45; color: white;}
    .gt-card.muted {border-color: #d5dddd; background: #f4f7f7;}
    .gt-card.muted .value {color: #5c6d71;}
    .gt-card.muted .gt-tag {background: #5c6d71; color: white;}
    .gt-card .gt-tag {background: #0d6b45; color: white;}
    .gt-note {background: white; border-radius: 12px; border: 1px solid #e2e8e8;
              padding: 0.4rem 0.75rem; margin-bottom: 0.3rem; color: #102124; font-size: 0.92rem;}
    .gt-change {background: #f3fafa; border-radius: 12px; border: 1px solid #e2e8e8;
                border-left: 4px solid #0F6E6E; padding: 0.45rem 0.75rem; margin-top: 0.15rem;
                color: #0F6E6E; font-size: 0.95rem; font-weight: 650;}
    .gt-banner {border-radius: 12px; padding: 0.4rem 0.75rem; margin: 0.2rem 0 0.35rem;}
    .gt-banner.danger {background: #fff6f5; border: 1px solid #f0c3bd;}
    .gt-banner.ok {background: #f4fbf7; border: 1px solid #c6e6d4;}
    .gt-banner.warn {background: #fffaf2; border: 1px solid #f3d7a8;}
    .gt-banner strong {font-size: 1.05rem;}
    .gt-foot {color: #2c4246; font-size: 0.9rem; font-weight: 600; margin-top: 0.7rem;}
    .gt-guide {color: #2c4246; font-size: 0.95rem; margin: 0.2rem 0 0.55rem;}
    .gt-badge {display: inline-block; background: #0F6E6E; color: white; border-radius: 999px;
               padding: 0.22rem 0.7rem; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em;
               margin-bottom: 0.45rem;}
    .gt-scenario {display: inline-block; background: #d5ebe9; color: #102124; font-size: 1.02rem;
                  font-weight: 750; letter-spacing: 0.01em; border-radius: 999px;
                  padding: 0.32rem 0.85rem; margin: 0.15rem 0 0.35rem;}
    .gt-try {color: #3d5156; font-size: 0.98rem; font-weight: 400; margin: 0.05rem 0 0.3rem;}
    .gt-q {position: relative; display: inline-flex; align-items: center; justify-content: center;
           width: 0.95rem; height: 0.95rem; margin-left: 0.28rem; border-radius: 999px;
           background: #0F6E6E; color: white; font-size: 0.68rem; font-weight: 700;
           letter-spacing: 0; text-transform: none; cursor: help; vertical-align: 0.08rem;}
    .gt-q::after {content: attr(data-tip); display: none; position: absolute; z-index: 40;
                  top: 1.25rem; left: 0; width: 250px; padding: 0.45rem 0.6rem;
                  background: #102124; color: white; border-radius: 8px;
                  font-size: 0.8rem; font-weight: 500; line-height: 1.35;
                  letter-spacing: 0; text-transform: none; white-space: normal;
                  box-shadow: 0 8px 20px rgba(16,33,36,0.22); pointer-events: none;}
    .gt-q.end::after {left: auto; right: 0;}
    .gt-q:hover::after, .gt-q:focus::after {display: block;}
    .gt-badge .gt-q, .gt-step.active .gt-q {background: white; color: #0F6E6E;}
    [data-testid="stMarkdown"], [data-testid="stMarkdownContainer"] {overflow: visible;}
    h4 {margin: 0.15rem 0 0.25rem !important; font-size: 1.05rem !important;}
    .gt-step {border: 1px solid #c5d8d6; border-radius: 10px; padding: 0.45rem 0.65rem;
              margin-bottom: 0.4rem; color: #3d5156; font-weight: 650;}
    .gt-step.active {background: #e7f4f3; border-color: #0F6E6E; color: #0F6E6E;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #2c4246 !important; font-size: 0.95rem !important; font-weight: 550;
    }
    [data-testid="stExpander"] summary p {color: #102124 !important; font-weight: 650;}
    div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
    div[data-stale="true"],
    [data-stale="true"],
    [data-stale="true"] * {
        opacity: 1 !important;
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_LABEL = {
    "Message them": "⚠ Message them",
    "Low sugar likely": "▼ Low sugar likely",
    "Sugar already too high": "▲ Sugar already too high",
    "Keep an eye": "· Keep an eye",
    "Okay for now": "✓ Okay for now",
}

WHATIF_LABEL = {
    "walk": "a 30-minute walk",
    "meal": "a meal 90 minutes later",
    "sleep": "a full night of sleep",
    "custom": "custom numbers",
}


def _q(tip: str, end: bool = False) -> str:
    """A hover question mark. The tip is the guide for that label."""
    safe = html.escape(tip, quote=True)
    klass = "gt-q end" if end else "gt-q"
    return f"<span class='{klass}' tabindex='0' aria-label='{safe}' data-tip=\"{safe}\">?</span>"


def _card(
    label: str,
    value: str,
    hint: str,
    tone: str = "",
    old: str | None = None,
    tag: str = "",
    tip: str = "",
    tip_end: bool = False,
) -> str:
    klass = f"gt-card {tone}".strip()
    tag_html = f"<span class='gt-tag'>{tag}</span>" if tag else ""
    mark = _q(tip, tip_end) if tip else ""
    old_html = f"<div class='old'>{old}</div>" if old else ""
    return (
        f"<div class='{klass}'><div class='label'>{label} {tag_html}{mark}</div>"
        f"<div class='value'>{value}</div>{old_html}"
        f"<div class='hint'>{hint}</div></div>"
    )


def _tone_glucose(value: float) -> str:
    if value >= 180 or value <= 70:
        return "danger"
    if value >= 140:
        return "warn"
    return "ok"


def _tone_risk(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "danger"
    if prob >= 0.35:
        return "warn"
    return "ok"


def _risk_tag(tone: str) -> str:
    return {"danger": "High", "warn": "Watch", "ok": "Okay"}.get(tone, "")


def _status(prob: float, glucose: float, threshold: float, hypo: float) -> str:
    if glucose >= 180:
        return "Sugar already too high"
    if hypo >= 0.20:
        return "Low sugar likely"
    if glucose < 150 and prob >= threshold:
        return "Message them"
    if prob >= 0.35 or glucose >= 140:
        return "Keep an eye"
    return "Okay for now"


def _article(n: int) -> str:
    if n in {8, 11, 18} or 80 <= n <= 89:
        return "an"
    return "a"


def _headline(
    prob: float,
    hypo: float,
    glucose: float,
    when: pd.Timestamp,
    threshold: float,
    was: int | None = None,
    delta_30m: float = 0.0,
) -> tuple[str, str]:
    until = (when + pd.Timedelta(hours=2)).strftime("%H:%M")
    chance = int(round(prob * 100))
    hypo_chance = int(round(hypo * 100))
    was_text = f" (was {was}%)" if was is not None else ""
    if prob >= threshold:
        return (
            "danger",
            f"Sugar is {glucose:.0f} now, but there is {_article(chance)} {chance}% chance it "
            f"passes 180 before {until}{was_text}. This is a warning, not a diagnosis.",
        )
    if hypo >= 0.20:
        tone = "danger" if hypo >= 0.5 or glucose <= 70 else "warn"
        movement = " and falling" if delta_30m < 0 else ", low and close to the 70 line"
        return (
            tone,
            f"Sugar is {glucose:.0f} now{movement}. There is a {hypo_chance}% chance it "
            f"drops under 70 before {until}. This is a warning, not a diagnosis.",
        )
    if prob < 0.20 and hypo < 0.20:
        return ("ok", f"Sugar is {glucose:.0f} now. No warning right now.")
    return (
        "ok",
        f"Sugar is {glucose:.0f} now. Nothing here says it will pass 180 before {until}{was_text}. "
        "This is a warning, not a diagnosis.",
    )


def _display_id(patient_id: str) -> str:
    digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
    return f"P-{digits}" if digits else str(patient_id)


def _whatif_on(extra_steps: float, meal_delay: float, sleep_choice: float, recorded_sleep: float) -> bool:
    return extra_steps > 0 or meal_delay > 0 or abs(sleep_choice - recorded_sleep) >= 0.15


def _whatif_preset(extra_steps: float, meal_delay: float, sleep_choice: float, recorded_sleep: float) -> str | None:
    if not _whatif_on(extra_steps, meal_delay, sleep_choice, recorded_sleep):
        return None
    sleep_unchanged = abs(sleep_choice - recorded_sleep) < 0.15
    if extra_steps == 3000 and meal_delay == 0 and sleep_unchanged:
        return "walk"
    if extra_steps == 0 and meal_delay == 90 and sleep_unchanged:
        return "meal"
    if extra_steps == 0 and meal_delay == 0 and abs(sleep_choice - 8.0) < 0.15:
        return "sleep"
    return "custom"


@st.cache_data(show_spinner=False)
def _load():
    load_bundle.cache_clear()
    return load_bundle()


def _ensure_artifacts() -> None:
    if artifacts_ready():
        return
    with st.spinner("Preparing the clinic list. This takes about a minute the first time."):
        train()
    load_bundle.cache_clear()
    _load.clear()


def _mark_peak(fig: go.Figure, path: pd.DataFrame, label: float, color: str, yshift: int) -> None:
    values = path["predicted_glucose"].to_numpy()
    idx = int(np.argmax(values))
    fig.add_annotation(
        x=path["timestamp"].iloc[idx],
        y=float(values[idx]),
        text=f"{label:.0f}",
        showarrow=False,
        yshift=yshift,
        font=dict(size=14, color=color),
        bgcolor="rgba(255,255,255,0.9)",
    )


def _chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    now: pd.Timestamp,
    current: float,
    show_outcome: bool,
    whatif_forecast: pd.DataFrame | None = None,
    original_peak: float | None = None,
    changed_peak: float | None = None,
    hide_forecast: bool = False,
) -> go.Figure:
    observed = history[history["timestamp"] <= now]
    future = history[history["timestamp"] > now]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor="rgba(15,110,110,0.08)", line_width=0)
    fig.add_hline(
        y=180,
        line_dash="dot",
        line_color="#9f2a1f",
        annotation_text="High 180",
        annotation_position="top left",
    )
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#8a5a08",
        annotation_text="Low 70",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=now,
        line_color="#102124",
        line_width=1.6,
        annotation_text="Now",
        annotation_position="top right",
    )
    fig.add_trace(
        go.Scatter(
            x=observed["timestamp"],
            y=observed["glucose_mgdl"],
            name="So far",
            line=dict(color="#1B7F4E", width=2.6),
            hovertemplate="So far<br>%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<br>Recorded before this minute<extra></extra>",
        )
    )
    if show_outcome and not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future["timestamp"],
                y=future["glucose_mgdl"],
                name="What happened next",
                line=dict(color="#9aa8ab", width=1.6, dash="dot"),
                hovertemplate="What happened next<br>%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<br>Recorded later. Demo only.<extra></extra>",
            )
        )
    if hide_forecast:
        pass
    elif whatif_forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=list(forecast["timestamp"]) + list(whatif_forecast["timestamp"][::-1]),
                y=list(forecast["predicted_glucose"]) + list(whatif_forecast["predicted_glucose"][::-1]),
                fill="toself",
                fillcolor="rgba(181, 71, 8, 0.16)",
                line=dict(width=0),
                hoverinfo="skip",
                name="Change from this what-if",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["predicted_glucose"],
                line=dict(color="#b7c2c4", width=1.4, dash="dash"),
                showlegend=False,
                hovertemplate="Original peak sketch<br>%{y:.0f} mg/dL<br>The guess before this what-if<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=whatif_forecast["timestamp"],
                y=whatif_forecast["predicted_glucose"],
                name="After this change",
                line=dict(color="#b54708", width=3.2, dash="dash"),
                hovertemplate="After this change<br>%{y:.0f} mg/dL<br>A new guess, not a new sensor reading<extra></extra>",
            )
        )
        _mark_peak(fig, forecast, original_peak if original_peak is not None else float(forecast["predicted_glucose"].max()), "#7d8b8e", 16)
        _mark_peak(
            fig,
            whatif_forecast,
            changed_peak if changed_peak is not None else float(whatif_forecast["predicted_glucose"].max()),
            "#b54708",
            -16,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["predicted_glucose"],
                line=dict(color="#b54708", width=2.4, dash="dash"),
                showlegend=False,
                hovertemplate="Peak sketch<br>%{y:.0f} mg/dL<br>One guess for the high in the next 2 hours<extra></extra>",
            )
        )
        if original_peak is not None:
            _mark_peak(fig, forecast, original_peak, "#b54708", 14)
    fig.add_trace(
        go.Scatter(
            x=[now],
            y=[current],
            name="Now",
            mode="markers",
            marker=dict(size=11, color="#102124"),
            showlegend=False,
            hovertemplate="Now<br>%{y:.0f} mg/dL<br>The reading this page is scored on<extra></extra>",
        )
    )
    fig.update_layout(
        height=390,
        margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(orientation="h", y=1.14, font=dict(size=12, color="#102124")),
        yaxis_title="Blood sugar (mg/dL)",
        xaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(color="#102124"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, range=[60, 280 if show_outcome else 260])
    return fig


def _open_person(patient_id: str, when: pd.Timestamp) -> None:
    st.session_state.screen = "person"
    st.session_state.patient_id = patient_id
    st.session_state.when = pd.Timestamp(when).isoformat()
    st.session_state.reset_whatif = True


def _ward_frame(roster: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = roster.copy()
    frame["status"] = [
        _status(float(prob), float(glucose), threshold, float(hypo))
        for prob, glucose, hypo in zip(frame["hyper_proba"], frame["glucose_mgdl"], frame["hypo_proba"])
    ]
    # Spike warnings and low-sugar warnings share the top band, ordered by
    # whichever of the two scores is higher. People already over 180 follow.
    band = {
        "Low sugar likely": 0,
        "Message them": 0,
        "Sugar already too high": 1,
        "Keep an eye": 2,
        "Okay for now": 3,
    }
    frame["_order"] = frame["status"].map(band)
    frame["_risk"] = frame[["hyper_proba", "hypo_proba"]].max(axis=1)
    frame = frame.sort_values(["_order", "_risk"], ascending=[True, False])
    frame["ID"] = frame["patient_id"].map(_display_id)
    frame["Who"] = frame["name"]
    frame["Age"] = frame["age"].astype(int)
    frame["City"] = frame["city"]
    frame["Condition"] = frame["glycemic_status"]
    frame["Blood sugar now"] = frame["glucose_mgdl"].round(0).astype(int)
    frame["Chance it shoots up"] = (frame["hyper_proba"] * 100).round(0).astype(int)
    labels = []
    for status, pct, hypo in zip(frame["status"], frame["Chance it shoots up"], frame["hypo_proba"]):
        if status == "Message them":
            labels.append(f"{STATUS_LABEL[status]} · {'High' if int(pct) >= 80 else 'Medium'}")
        elif status == "Low sugar likely":
            labels.append(f"{STATUS_LABEL[status]} · {'High' if float(hypo) >= 0.5 else 'Watch'}")
        else:
            labels.append(STATUS_LABEL[status])
    frame["What to do"] = labels
    frame["Chance it drops too low"] = [
        "—" if float(p) < 0.20 else f"{float(p) * 100:.0f}%" for p in frame["hypo_proba"]
    ]
    return frame


def _nearest(times: list[pd.Timestamp], target: pd.Timestamp) -> int:
    return min(range(len(times)), key=lambda i: abs(times[i] - target))


def _clock(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.strftime('%a')} {int(ts.day)} {ts.strftime('%b')}, {ts.strftime('%H:%M')}"


def _meal_slot(when: pd.Timestamp, morning: pd.Timestamp, lunch: pd.Timestamp, dinner: pd.Timestamp) -> str | None:
    for name, target in (("Breakfast", morning), ("Lunch", lunch), ("Dinner", dinner)):
        if abs(pd.Timestamp(when) - pd.Timestamp(target)) <= pd.Timedelta(minutes=20):
            return name
    return None


def _meal_ok(when: pd.Timestamp) -> bool:
    hour = pd.Timestamp(when).hour + pd.Timestamp(when).minute / 60.0
    return 6.0 <= hour < 22.0


def _on_meal_pick() -> None:
    pick = st.session_state.get("meal_time")
    targets = st.session_state.get("gt_meal_targets") or {}
    times = st.session_state.get("gt_clinic_times") or []
    if not pick or pick not in targets or not times:
        return
    st.session_state.clinic_index = _nearest(times, pd.Timestamp(targets[pick]))


def _on_clinic_slider() -> None:
    times = st.session_state.get("gt_clinic_times") or []
    targets = st.session_state.get("gt_meal_targets") or {}
    if not times or "clinic_index" not in st.session_state:
        return
    when = pd.Timestamp(times[int(st.session_state.clinic_index)])
    st.session_state.meal_time = _meal_slot(
        when,
        pd.Timestamp(targets["Breakfast"]),
        pd.Timestamp(targets["Lunch"]),
        pd.Timestamp(targets["Dinner"]),
    )


WALK_LABEL = "Walk 30 min"
MEAL_LABEL = "Meal 90 min later"
SLEEP_LABEL = "Full night's sleep"


def _on_whatif_pick() -> None:
    pick = st.session_state.get("whatif_pick")
    recorded = float(st.session_state.get("gt_recorded_sleep", 6.2))
    if pick == WALK_LABEL:
        st.session_state.extra_steps = 3000
        st.session_state.meal_delay = 0
        st.session_state.sleep_choice = recorded
    elif pick == MEAL_LABEL:
        st.session_state.extra_steps = 0
        st.session_state.meal_delay = 90
        st.session_state.sleep_choice = recorded
    elif pick == SLEEP_LABEL:
        st.session_state.extra_steps = 0
        st.session_state.meal_delay = 0
        st.session_state.sleep_choice = 8.0
    else:
        st.session_state.extra_steps = 0
        st.session_state.meal_delay = 0
        st.session_state.sleep_choice = recorded


def _on_whatif_slider() -> None:
    recorded = float(st.session_state.get("gt_recorded_sleep", 6.2))
    extra = int(st.session_state.get("extra_steps", 0))
    meal = int(st.session_state.get("meal_delay", 0))
    sleep = float(st.session_state.get("sleep_choice", recorded))
    preset = _whatif_preset(extra, meal, sleep, recorded)
    label = {"walk": WALK_LABEL, "meal": MEAL_LABEL, "sleep": SLEEP_LABEL}.get(preset or "")
    if label == MEAL_LABEL and not st.session_state.get("gt_meal_ok", True):
        label = None
    st.session_state.whatif_pick = label


PLAIN_FEATURE = {
    "glucose_mgdl": "Blood sugar now",
    "glucose_delta_30m": "Change over 30 minutes",
    "glucose_mean_1h": "Average sugar this hour",
    "glucose_std_2h": "Sugar ups and downs",
    "glucose_min_6h": "Lowest sugar in 6 hours",
    "glucose_max_6h": "Highest sugar in 6 hours",
    "heart_rate_1h": "Heart rate",
    "hrv_6h": "Heart rhythm variation",
    "steps_2h": "Steps in the last 2 hours",
    "steps_24h": "Steps in the last day",
    "sleep_hours_prev": "Last night's sleep",
    "deep_sleep_min_prev": "Deep sleep",
    "hba1c": "3-month sugar average",
    "fasting_glucose": "Fasting sugar",
    "on_metformin": "Metformin",
    "on_insulin": "Insulin",
    "on_sglt2": "SGLT2 medicine",
    "family_history": "Family history",
    "tcf7l2_alleles": "Gene marker TCF7L2",
    "kcnj11_alleles": "Gene marker KCNJ11",
    "sbp": "Blood pressure",
    "ldl": "Cholesterol (LDL)",
    "egfr": "Kidney filter rate",
    "hypertension": "High blood pressure flag",
    "bmi": "Body weight for height",
    "age": "Age",
    "sex_male": "Sex",
    "diabetes_duration_years": "Years with diabetes",
}


def _reliance_chart(importance: pd.DataFrame) -> go.Figure:
    grouped: dict[str, float] = {}
    time_of_day = 0.0
    for feature, drop in zip(importance["feature"], importance["auc_drop"]):
        value = float(drop)
        if feature in {"hour_sin", "hour_cos"}:
            time_of_day += value
            continue
        name = PLAIN_FEATURE.get(str(feature), str(feature))
        grouped[name] = grouped.get(name, 0.0) + value
    grouped["Time of day"] = time_of_day
    ranked = sorted(((name, value) for name, value in grouped.items() if value > 0), key=lambda item: item[1])
    truncated = len(ranked) > 8
    top = ranked[-8:]
    names = [name for name, _ in top]
    values = [value for _, value in top]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#0F6E6E",
            hovertemplate="%{y}<br>Drop in ROC-AUC: %{x:.3f}<br>A longer bar means the model used this more on practice data.<extra></extra>",
        )
    )
    ceiling = max(values) * 1.12 if values else 1
    title = (
        "Top 8 inputs the model relied on (practice data)"
        if truncated
        else "What the model relied on (practice data)"
    )
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=8, r=8, t=48, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#102124"),
        xaxis_title="Drop in ROC-AUC when this is removed",
        yaxis_title=None,
    )
    fig.update_xaxes(range=[0, ceiling], showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _today(bundle: dict, times: list[pd.Timestamp], threshold: float) -> None:
    st.markdown(
        "<div class='gt-kicker'>Clinic list "
        + _q("Step 1. One minute for every person still in the practice set. Open a chart only when a row needs it.")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='gt-title'>Who should get a message? "
        + _q("People whose sugar is still ordinary, but may pass 180 in the next 2 hours, are listed first.")
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("What the labels mean"):
        st.markdown(
            "**Message them** means sugar looks okay now, but it may pass 180 in the next 2 hours. "
            "**Low sugar likely** means the chance of dropping under 70 is 20% or more, "
            "and sugar is not already over 180. "
            "**Sugar already too high** means it is already over 180. "
            "Warnings are listed first, ordered by whichever of the two risks is higher. "
            "Press **Step 2**, or choose a name and press **Open**."
        )
    morning = pd.Timestamp(bundle["moments"]["spike"]["timestamp"])
    lunch = morning.normalize() + pd.Timedelta(hours=13, minutes=30)
    dinner = morning.normalize() + pd.Timedelta(hours=20)
    when = times[st.session_state.clinic_index]
    roster = snapshot_ward(bundle, when)
    ranked = _ward_frame(roster, threshold)
    calls = int((ranked["status"] == "Message them").sum())
    low_rows = ranked[ranked["status"] == "Low sugar likely"]
    low = int(len(low_rows))
    high = int((ranked["status"] == "Sugar already too high").sum())
    quiet = int(ranked["status"].isin(["Keep an eye", "Okay for now"]).sum())
    low_severe = bool((low_rows["hypo_proba"] >= 0.5).any()) if low else False
    low_tone = "danger" if low_severe else "warn" if low else "ok"
    low_tag = "High" if low_severe else "Watch" if low else "Okay"
    a, b, c, d = st.columns(4)
    a.markdown(
        _card(
            "Message them",
            str(calls),
            "Sugar looks okay, but a rise is likely",
            "danger" if calls else "ok",
            tag="High" if calls else "Okay",
            tip="Sugar is under 150, and the chance of passing 180 in the next 2 hours is at least 58%.",
        ),
        unsafe_allow_html=True,
    )
    b.markdown(
        _card(
            "Low sugar likely",
            str(low),
            "May drop under 70 in the next 2 hours",
            low_tone,
            tag=low_tag,
            tip="Not already over 180, and the chance of dropping under 70 is 20% or more.",
        ),
        unsafe_allow_html=True,
    )
    c.markdown(
        _card(
            "Already too high",
            str(high),
            "Sugar is already over 180",
            "warn" if high else "ok",
            tag="Watch" if high else "Okay",
            tip="The reading is already at or above 180. The twin is not needed to notice that.",
        ),
        unsafe_allow_html=True,
    )
    d.markdown(
        _card(
            "No message",
            str(quiet),
            "Keep an eye, or they are okay",
            "ok",
            tag="Okay",
            tip="Everyone else at this minute. Keep an eye, or the row looks okay.",
            tip_end=True,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='gt-guide'>This list is for {when.strftime('%A %d %B, %H:%M')}. "
        + _q("Change the minute with Breakfast, Lunch, Dinner, or the slider. The four counts rebuild for that minute.")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.session_state.gt_clinic_times = times
    st.session_state.gt_meal_targets = {
        "Breakfast": pd.Timestamp(morning).isoformat(),
        "Lunch": pd.Timestamp(lunch).isoformat(),
        "Dinner": pd.Timestamp(dinner).isoformat(),
    }
    if "meal_time" not in st.session_state:
        st.session_state.meal_time = _meal_slot(when, morning, lunch, dinner)
    st.segmented_control(
        "Clinic time",
        ["Breakfast", "Lunch", "Dinner"],
        key="meal_time",
        on_change=_on_meal_pick,
        label_visibility="collapsed",
        help="Breakfast is Wednesday 9 September at 08:00. Lunch is 13:30. Dinner is 20:00. These are bookmarks on the same day.",
    )
    st.select_slider(
        "Or drag to any time",
        options=list(range(len(times))),
        format_func=lambda i: _clock(times[i]),
        key="clinic_index",
        on_change=_on_clinic_slider,
        help="Any 15-minute slot from Tuesday 1 September through Saturday 12 September 2026. The list is that one minute for every person.",
    )
    first = ranked.iloc[0]
    if st.button(
        f"Step 2: open {first['Who']}",
        type="primary",
        help="Opens the person at the top of this list, at the time on the slider.",
    ):
        _open_person(first["patient_id"], when)
        st.rerun()

    query = st.text_input(
        "Or search a name, city, or ID",
        placeholder="Try Mumbai, or a first name",
        help="Filters the table only. The four counts above stay for the whole clinic at this minute.",
    )
    if query.strip():
        needle = query.strip().lower()
        ranked = ranked[
            ranked["Who"].str.lower().str.contains(needle, na=False)
            | ranked["City"].str.lower().str.contains(needle, na=False)
            | ranked["ID"].str.lower().str.contains(needle, na=False)
            | ranked["patient_id"].str.lower().str.contains(needle, na=False)
        ]
    if ranked.empty:
        st.info("No one matches that search.")
        return
    shown = ranked[
        [
            "ID",
            "Who",
            "Age",
            "City",
            "Condition",
            "What to do",
            "Blood sugar now",
            "Chance it shoots up",
            "Chance it drops too low",
        ]
    ].reset_index(drop=True)
    row_height = 35
    visible_rows = min(len(shown), 10)
    painted = shown.style.map(_paint_action, subset=["What to do"])
    st.dataframe(
        painted,
        width="stretch",
        height=36 + visible_rows * row_height,
        row_height=row_height,
        hide_index=True,
        key="ward-table",
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small", help="Practice ID for this demo. P-140 is Reyansh. P-080 is Kiara Menon, age 52."),
            "Who": st.column_config.TextColumn("Who", width=220, help="Generated name. Several people can share a name. Use the ID."),
            "Age": st.column_config.NumberColumn("Age", help="Age in years. The cohort is 30 to 55."),
            "City": st.column_config.TextColumn("City", help="City on the synthetic chart."),
            "Condition": st.column_config.TextColumn("Condition", help="Type 2 diabetes or prediabetes, as written on the chart."),
            "What to do": st.column_config.TextColumn(
                "What to do",
                width=220,
                help="Message them, low sugar likely, already too high, keep an eye, or okay for now. High means 80% or more for a rise, or 50% or more for a drop.",
            ),
            "Blood sugar now": st.column_config.NumberColumn("Blood sugar now", help="Sensor reading at this minute, in mg/dL."),
            "Chance it shoots up": st.column_config.ProgressColumn(
                "Chance it shoots up",
                help="Chance the highest sugar in the next 2 hours reaches 180 or more.",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "Chance it drops too low": st.column_config.TextColumn(
                "Chance it drops too low",
                help="Shown when the chance of dropping under 70 is 20% or more. A dash means that chance is under 20%.",
            ),
        },
    )
    pick_l, pick_r = st.columns([5, 1], vertical_alignment="bottom")
    open_id = pick_l.selectbox(
        "Or open someone else",
        options=list(ranked["patient_id"]),
        format_func=lambda pid: f"{ranked.loc[ranked['patient_id'] == pid, 'Who'].iloc[0]} · {_display_id(pid)}",
        help="Anyone on this list, at the same clinic time. This does not change the ranking.",
    )
    if pick_r.button("Open", type="secondary", help="Opens the person chosen in the box."):
        _open_person(open_id, when)
        st.rerun()


def _bookmark_on(patient_id: str, when: pd.Timestamp, bookmark: dict | None) -> bool:
    if not bookmark:
        return False
    return patient_id == bookmark["patient_id"] and abs(pd.Timestamp(when) - pd.Timestamp(bookmark["timestamp"])) <= pd.Timedelta(minutes=1)


def _rerun_panel() -> None:
    """Rerun the what-if panel. A full-page run cannot use a fragment-scoped rerun."""
    try:
        st.rerun(scope="fragment")
    except StreamlitInvalidLayoutContextError:
        st.rerun()


def _clear_whatif(recorded_sleep: float) -> None:
    st.session_state.extra_steps = 0
    st.session_state.meal_delay = 0
    st.session_state.sleep_choice = float(recorded_sleep)
    st.session_state.whatif_pick = None
    st.session_state.show_24h = False
    st.session_state.show_outcome = False


def _glucose_hours_ago(history: pd.DataFrame, when: pd.Timestamp, hours: float = 2.0) -> float | None:
    """Glucose on the 15-minute grid at this many hours before now."""
    target = pd.Timestamp(when) - pd.Timedelta(hours=hours)
    past = history[history["timestamp"] <= pd.Timestamp(when)]
    if past.empty:
        return None
    idx = (past["timestamp"] - target).abs().idxmin()
    hit = past.loc[idx]
    if abs(pd.Timestamp(hit["timestamp"]) - target) > pd.Timedelta(minutes=20):
        return None
    return float(hit["glucose_mgdl"])


def _paint_action(value: str) -> str:
    text = str(value)
    if "Low sugar likely" not in text:
        return ""
    if "High" in text:
        return "background-color: #fff6f5; color: #9f2a1f; font-weight: 700"
    return "background-color: #fffaf2; color: #8a5a08; font-weight: 700"


@st.fragment
def _whatif_panel(patient_id: str, when_iso: str) -> None:
    bundle = _load()
    when = pd.Timestamp(when_iso)
    view = view_model(bundle, patient_id, when)
    row = view["row"]
    prob = float(row["hyper_proba"])
    glucose = float(row["glucose_mgdl"])
    peak = float(row["predicted_peak"])
    hypo = float(row["hypo_proba"])
    recorded_sleep = float(np.clip(row["sleep_hours_prev"], 4.0, 9.0))
    meal_ok = _meal_ok(when)
    st.session_state.gt_recorded_sleep = recorded_sleep
    st.session_state.gt_meal_ok = meal_ok

    if st.session_state.pop("reset_whatif", False) or st.session_state.pop("clear_selection", False):
        _clear_whatif(recorded_sleep)
    if not meal_ok:
        if int(st.session_state.get("meal_delay", 0)) != 0:
            st.session_state.meal_delay = 0
        if st.session_state.get("whatif_pick") == MEAL_LABEL:
            st.session_state.whatif_pick = None

    extra_steps = int(st.session_state.get("extra_steps", 0))
    meal_delay = int(st.session_state.get("meal_delay", 0))
    sleep_choice = float(st.session_state.get("sleep_choice", recorded_sleep))
    scenario = what_if(row, extra_steps, sleep_choice, meal_delay)
    active = _whatif_on(extra_steps, meal_delay, sleep_choice, recorded_sleep)
    preset = _whatif_preset(extra_steps, meal_delay, sleep_choice, recorded_sleep)
    shown_prob = float(scenario["hyper_proba"]) if active else prob
    shown_peak = float(scenario["predicted_peak"]) if active else peak
    shown_hypo = float(scenario["hypo_proba"]) if active else hypo

    if preset:
        st.markdown(
            f"<div class='gt-scenario'>With {WHATIF_LABEL[preset]} (model estimate) "
            + _q("The tiles below are a new score for this same minute. The green line on the chart does not move.")
            + "</div>",
            unsafe_allow_html=True,
        )
    was = int(round(prob * 100)) if active else None
    tone, sentence = _headline(
        shown_prob,
        shown_hypo,
        glucose,
        when,
        view["threshold"],
        was=was,
        delta_30m=float(row["glucose_delta_30m"]),
    )
    st.markdown(
        f"<div class='gt-banner {tone}'><strong>{sentence}</strong> "
        + _q("The sentence for this minute. A warning is not a diagnosis, and it is not an order to change a dose.")
        + "</div>",
        unsafe_allow_html=True,
    )

    if shown_hypo >= 0.5:
        hypo_tone, hypo_tag = "danger", "High"
    elif shown_hypo >= 0.20:
        hypo_tone, hypo_tag = "warn", "Watch"
    else:
        hypo_tone, hypo_tag = "ok", "Okay"
    low_case = shown_hypo >= 0.20
    risk_tone = _tone_risk(shown_prob, view["threshold"])
    peak_tone = _tone_glucose(shown_peak)
    sugar_tone = _tone_glucose(glucose)
    if low_case and sugar_tone == "ok":
        sugar_tone = "warn"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _card(
            "Blood sugar now",
            f"{glucose:.0f} mg/dL",
            "Usually okay under 140",
            sugar_tone,
            tag=_risk_tag(sugar_tone),
            tip="The sensor reading at this minute. Under 70 is too low. 180 or above is already too high.",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _card(
            "Chance it shoots up",
            f"{shown_prob * 100:.0f}%",
            "In the next 2 hours",
            risk_tone,
            old=f"{prob * 100:.0f}%" if active and abs(shown_prob - prob) >= 0.005 else None,
            tag=_risk_tag(risk_tone),
            tip="Chance the highest sugar in the next 2 hours reaches 180. A crossed-out number is the score before this what-if.",
        ),
        unsafe_allow_html=True,
    )
    if low_case:
        c3.markdown(
            _card(
                "How low it may get",
                "—",
                "The model does not estimate a low point",
                "muted",
                tag="No estimate",
                tip="There is no guess for a lowest number. The red tile beside this one is the chance of dropping under 70.",
            ),
            unsafe_allow_html=True,
        )
    else:
        c3.markdown(
            _card(
                "How high it may get",
                f"{shown_peak:.0f}",
                "The guess for the next 2 hours",
                peak_tone,
                old=f"{peak:.0f}" if active and abs(shown_peak - peak) >= 0.5 else None,
                tag=_risk_tag(peak_tone),
                tip="One number for the highest sugar in the next 2 hours, in mg/dL. It is a guess, not a range.",
            ),
            unsafe_allow_html=True,
        )
    c4.markdown(
        _card(
            "Chance it drops too low",
            f"{shown_hypo * 100:.0f}%",
            "Under 70 is too low",
            hypo_tone,
            old=f"{hypo * 100:.0f}%" if active and abs(shown_hypo - hypo) >= 0.01 else None,
            tag=hypo_tag,
            tip="A separate score, on the same chart and sensors, for sugar falling to 70 or below. 20% or more is a warning.",
            tip_end=True,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4>Try a what-if "
        + _q("A question for the doctor. Walk adds 3,000 steps. A later meal moves the clock. Sleep sets last night to 8 hours. Nothing here is a prescription.")
        + "</h4>",
        unsafe_allow_html=True,
    )
    options = [WALK_LABEL, SLEEP_LABEL]
    if meal_ok:
        options = [WALK_LABEL, MEAL_LABEL, SLEEP_LABEL]
    ctrl, reset = st.columns([4.2, 1], vertical_alignment="bottom")
    ctrl.segmented_control(
        "Try a what-if",
        options,
        key="whatif_pick",
        on_change=_on_whatif_pick,
        label_visibility="collapsed",
        help="Walk adds 3,000 steps. Meal delay is 90 minutes and is hidden at night. Sleep sets last night to 8 hours. Only one of these is on at a time.",
    )
    if reset.button("Reset", help="Clears the what-if and restores the original scores for this minute."):
        st.session_state.clear_selection = True
        _rerun_panel()
    if not active:
        st.markdown(
            "<div class='gt-try'>Press a button to try a change. "
            + _q("The orange line is a sketch of the high. The green line is what the sensor already recorded.")
            + "</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.65, 1])
    with left:
        t1, t2 = st.columns([1, 2.2])
        show_24h = bool(st.session_state.get("show_24h", False))
        show_outcome = bool(st.session_state.get("show_outcome", False))
        if t1.button(
            "Show 24 hours",
            type="primary" if show_24h else "secondary",
            key="btn_24",
            help="Widens the green line from about 6 hours to 24 hours. The score does not change.",
        ):
            st.session_state.show_24h = not show_24h
            _rerun_panel()
        if t2.button(
            "Demo only: show what really happened",
            type="primary" if show_outcome else "secondary",
            key="btn_demo",
            help="Draws the glucose recorded after this minute, as a grey dotted line. Turn it off for the clinic view.",
        ):
            st.session_state.show_outcome = not show_outcome
            _rerun_panel()
        hours = 24 if show_24h else 6
        history = history_window(bundle, patient_id, pd.Timestamp(row["timestamp"]), hours=hours)
        whatif_forecast = None
        if active:
            alt = row.copy()
            alt["predicted_peak"] = shown_peak
            whatif_forecast = forecast_path(alt)
        st.plotly_chart(
            _chart(
                history,
                view["forecast"],
                pd.Timestamp(row["timestamp"]),
                glucose,
                show_outcome,
                whatif_forecast,
                original_peak=peak,
                changed_peak=shown_peak if active else None,
                hide_forecast=low_case,
            ),
            width="stretch",
        )
        if low_case:
            st.markdown(
                "<div class='gt-guide'>The forecast line shows highs only. This warning comes from the low-sugar score. "
                + _q("The orange sketch is for a spike. On this page the warning is the chance of a drop, so that sketch is hidden.")
                + "</div>",
                unsafe_allow_html=True,
            )
    with right:
        heading = "What the chart shows" if low_case else "Why this warning"
        heading_tip = (
            "The fall over the last 2 hours. On a low-sugar page the acceptable-range line and the blood-pressure line are left off."
            if low_case
            else "Facts from the chart and the watch that sit behind the score. They are not an instruction."
        )
        st.markdown(f"<h4>{heading} {_q(heading_tip, end=True)}</h4>", unsafe_allow_html=True)
        reasons = explain_row(row)
        if low_case:
            reasons = [
                text
                for text in reasons
                if "clinics treat as acceptable" not in text
                and "Blood pressure is high on the chart" not in text
            ]
            earlier = _glucose_hours_ago(history, pd.Timestamp(row["timestamp"]))
            if earlier is not None and earlier > glucose:
                reasons.insert(
                    0,
                    f"Sugar has fallen from about {earlier:.0f} to {glucose:.0f} mg/dL over the last 2 hours.",
                )
                reasons = reasons[:5]
        change_note = None
        if active:
            if preset == "walk":
                change_note = f"With the walk: about {extra_steps:,} more steps."
            elif preset == "meal":
                change_note = "With the meal 90 minutes later."
            elif preset == "sleep":
                change_note = f"With a full night's sleep: {sleep_choice:.0f} hours."
            else:
                change_note = "With the numbers in the section below."
        for reason in reasons:
            st.markdown(
                f"<div class='gt-note'>{html.escape(reason)} {_q('A fact from this chart or this watch. It explains the score. It does not tell you what to prescribe.', end=True)}</div>",
                unsafe_allow_html=True,
            )
        if change_note:
            st.markdown(f"<div class='gt-change'>{change_note}</div>", unsafe_allow_html=True)

    with st.expander("Or move the numbers yourself"):
        q1, q2, q3 = st.columns(3)
        q1.slider(
            "Extra steps",
            0,
            4000,
            step=200,
            key="extra_steps",
            on_change=_on_whatif_slider,
            help="Adds steps to the last day and re-scores. 3,000 is the same as Walk 30 min.",
        )
        q2.slider(
            "Hours of sleep last night",
            4.0,
            9.0,
            step=0.1,
            key="sleep_choice",
            on_change=_on_whatif_slider,
            help="Replaces last night's sleep and re-scores. 8 hours is the same as Full night's sleep.",
        )
        if meal_ok:
            q3.slider(
                "Delay the meal, minutes",
                0,
                120,
                step=15,
                key="meal_delay",
                on_change=_on_whatif_slider,
                help="Moves the time of day the model sees, as if the meal were this many minutes later. 90 is the meal button.",
            )
        else:
            q3.markdown(
                "<div class='gt-guide'>Meal delay is hidden at night. "
                + _q("From 22:00 to 06:00 the meal control stays off, because a delayed meal is a daytime question.")
                + "</div>",
                unsafe_allow_html=True,
            )


def _person(bundle: dict, roster: pd.DataFrame, moments: dict, metrics: dict) -> None:
    patient_id = st.session_state.patient_id
    times = list(bundle["scored"].loc[bundle["scored"]["patient_id"] == patient_id, "timestamp"].sort_values())
    target = pd.Timestamp(st.session_state.when)
    start_index = min(range(len(times)), key=lambda i: abs(times[i] - target))
    if st.session_state.get("moment_owner") != patient_id:
        st.session_state.moment_index = start_index
        st.session_state.moment_owner = patient_id
    chosen_index = int(st.session_state.get("moment_index", start_index))
    chosen_index = min(max(chosen_index, 0), len(times) - 1)
    when = pd.Timestamp(times[chosen_index])

    view = view_model(bundle, patient_id, when)
    row = view["row"]
    person = view["patient"]
    person_row = roster[roster["patient_id"] == patient_id]
    name = person_row.iloc[0]["name"] if not person_row.empty else person["name"]

    head_l, head_r = st.columns([0.7, 8], gap="small", vertical_alignment="center")
    if head_l.button("Back", help="Returns to the clinic list at the same minute."):
        st.session_state.screen = "today"
        st.rerun()
    head_r.markdown(
        f"<div class='gt-title'>{html.escape(str(name))} <span class='gt-when'>"
        f"· {when.strftime('%A, %d %B %Y · %H:%M')}</span> "
        + _q("Step 2. One person at one minute. The scores look forward 2 hours and stop at this reading.")
        + "</div>",
        unsafe_allow_html=True,
    )

    _whatif_panel(patient_id, when.isoformat())

    with st.expander("Health record and today's watch"):
        st.markdown(
            "<div class='gt-guide'>The clinic chart is on the left. The watch is on the right. "
            + _q("HbA1c is the 3-month sugar average. Blood pressure is context from the chart. It is not a prediction of a heart attack.")
            + "</div>",
            unsafe_allow_html=True,
        )
        record, watch = st.columns(2)
        with record:
            years = "just found" if float(person["diabetes_duration_years"]) < 0.5 else f"for {person['diabetes_duration_years']:.0f} years"
            if person.hypertension:
                pressure = f"Blood pressure is high: {person['sbp']:.0f} mmHg, systolic"
            else:
                pressure = f"Blood pressure is not flagged as high: {person['sbp']:.0f} mmHg, systolic"
            st.markdown(
                f"**{person['name']}**, age {int(person['age'])}, {person['sex']}, lives in {person['city']}  \n"
                f"{person['glycemic_status']} {years}  \n"
                f"3-month sugar average (HbA1c): {person['hba1c']:.1f}% "
                "(doctors usually want this under 7)  \n"
                f"{pressure}  \n"
                f"Medicines on file: "
                f"metformin {'yes' if person.on_metformin else 'no'}, "
                f"insulin {'yes' if person.on_insulin else 'no'}"
            )
        with watch:
            w1, w2 = st.columns(2)
            w1.metric(
                "Heart beats per minute",
                f"{row['heart_rate_1h']:.0f}",
                help="Average heart rate over the last hour. Context from the watch. Not a heart-attack forecast.",
            )
            w2.metric(
                "Steps in the last day",
                f"{row['steps_24h']:.0f}",
                help="Steps over the last 24 hours. The walk button adds 3,000 to the score, not to this recorded number.",
            )
            st.metric(
                "Sleep last night",
                f"{row['sleep_hours_prev']:.1f} hours",
                help="Hours in the previous completed night. The sleep button re-scores as if this were 8 hours.",
            )

    with st.expander("Examples and the technical review"):
        st.markdown(
            "<div class='gt-guide'>These two bookmarks are for a walkthrough. The list is the normal way in. "
            + _q("High-risk opens Reyansh at Wednesday 9 September 08:00. Quiet opens Aarav at Saturday 12 September 01:00.")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.select_slider(
            "Look at a different minute of this person's day",
            options=list(range(len(times))),
            format_func=lambda i: _clock(times[i]),
            key="moment_index",
            help="Another 15-minute reading for this same person. The clinic list stays on the minute you left.",
        )
        e1, e2 = st.columns(2)
        spike = moments.get("spike")
        calm = moments.get("calm")
        if spike and e1.button(
            "Open the high-risk example",
            type="primary" if _bookmark_on(patient_id, when, spike) else "secondary",
            help="Reyansh Joshi, Wednesday 9 September 2026, 08:00. Sugar is ordinary and a rise is likely.",
        ):
            _open_person(spike["patient_id"], spike["timestamp"])
            st.rerun()
        if calm and e2.button(
            "Open a quiet example",
            type="primary" if _bookmark_on(patient_id, when, calm) else "secondary",
            help="Aarav Joshi, Saturday 12 September 2026, 01:00. Both chances are under 20%, so there is no warning.",
        ):
            _open_person(calm["patient_id"], calm["timestamp"])
            st.rerun()
        fusion = metrics["fusion_test"]
        st.markdown(
            f"<div class='gt-guide'>ROC-AUC on the {int(metrics['n_test_patients'])} people the model had not seen: "
            f"{fusion['roc_auc']:.2f} for the full twin, and {metrics['cgm_only_test']['roc_auc']:.2f} "
            f"for the glucose sensor only. While sugar is still under 140, ROC-AUC is "
            f"{metrics.get('lead_time_auc_fusion', 0):.2f} and {metrics.get('lead_time_auc_cgm_only', 0):.2f}. "
            + _q("ROC-AUC is how well the score ranks people who do spike above people who do not. 0.50 is a coin toss. 1.00 is perfect. The second pair is the one that matters: sugar is still under 140, so the sensor alone has less to go on.")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_reliance_chart(bundle["importance"]), width="stretch")


def main() -> None:
    _ensure_artifacts()
    bundle = _load()
    times = clinic_times(bundle)
    threshold = float(bundle["metrics"]["threshold"])
    if "clinic_index" not in st.session_state:
        opening = pd.Timestamp(bundle["moments"]["spike"]["timestamp"])
        st.session_state.clinic_index = _nearest(times, opening)
    if "screen" not in st.session_state:
        st.session_state.screen = "today"
    roster = snapshot_ward(bundle, times[st.session_state.clinic_index])

    st.markdown(
        "<div class='gt-badge'>Practice data, not real patients "
        + _q("Every row is generated. This screen is a decision-support demo, not a medical device.")
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        step = 1 if st.session_state.screen == "today" else 2
        st.markdown(
            f"<div class='gt-step {'active' if step == 1 else ''}'>Step 1 of 2 · Clinic list "
            + _q("The ranked list for one minute. Start here.")
            + "</div>"
            f"<div class='gt-step {'active' if step == 2 else ''}'>Step 2 of 2 · One person "
            + _q("The chart, the two warnings, and a what-if for the person you opened.")
            + "</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.screen == "today":
        _today(bundle, times, threshold)
    else:
        _person(bundle, roster, bundle["moments"], bundle["metrics"])

    st.markdown(
        "<div class='gt-foot'>Practice data. Not real patients. Not medical advice. "
        + _q("Do not use these scores to change a dose. Hover a question mark, or the small help icon beside a control, to see what it does.")
        + "</div>",
        unsafe_allow_html=True,
    )


main()

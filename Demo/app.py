"""Streamlit Developer Dashboard for the AI Multi-Agent SDK Sandbox."""

from __future__ import annotations

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from sdk_core import NPCAgent


AGENT_DEFINITIONS = [
    (
        "Village Chief",
        "Calm and responsible. Prioritizes protecting villagers' lives and "
        "always gives defensive orders.",
    ),
]
FALLBACK_RESPONSE = (
    "I am organizing the village defense and evacuating civilians through the "
    "safest route. Stay calm and follow my orders!"
)


def configure_page() -> None:
    """Configure the dashboard and apply the dark developer-console styling."""
    st.set_page_config(
        page_title="AI Multi-Agent Orchestrator Sandbox",
        
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root { --panel: #151a22; --border: #2b3545; --muted: #8d9aae; }
        .block-container { max-width: 1440px; padding-top: 2rem; }
        .hero {
            background: linear-gradient(135deg, #162235 0%, #111722 100%);
            border: 1px solid #30445e; border-radius: 18px;
            padding: 1.5rem 1.8rem; margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0 0 .4rem; letter-spacing: -.03em; }
        .hero p { color: #aab7c8; margin: 0; }
        .agent-panel {
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 14px; padding: 1rem; min-height: 245px;
        }
        .agent-name { font-size: 1.2rem; font-weight: 700; margin-bottom: .35rem; }
        .agent-persona { color: var(--muted); line-height: 1.55; min-height: 72px; }
        .status { color: #62d49a; font-family: monospace; font-size: .78rem; }
        .response-box {
            background: #10151c; border-left: 3px solid #78b7ff;
            border-radius: 8px; padding: .8rem; margin-top: .9rem;
            font-family: "Cascadia Code", "Consolas", monospace;
        }
        .log-line {
            border-bottom: 1px solid #252d39; padding: .55rem 0;
            font-family: "Cascadia Code", "Consolas", monospace; font-size: .82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_agents(api_key: str) -> list[NPCAgent]:
    """Create the persistent NPC agent for the current Streamlit session."""
    return [NPCAgent(name, persona, api_key) for name, persona in AGENT_DEFINITIONS]


def run_agents_sequential(
    agents: list[NPCAgent],
    event: str,
) -> dict[str, str]:
    """Ask NPCs one at a time to avoid provider rate limits and unstable responses."""
    responses: dict[str, str] = {}
    for agent in agents:
        try:
            responses[agent.name] = agent.react(event)
        except Exception as error:
            responses[agent.name] = format_agent_error(str(error))
    return responses


def format_agent_error(error_text: str) -> str:
    """Turn transient provider errors into an actionable dashboard message."""
    upper_error = error_text.upper()
    if "503" in upper_error or "UNAVAILABLE" in upper_error:
        return "Gemini is temporarily busy. Please trigger the event again in a few seconds."
    if "429" in upper_error or "RESOURCE_EXHAUSTED" in upper_error:
        return "Gemini rate limit reached. Please wait briefly and trigger again."
    return f"Error: {error_text}"


def render_agent_panel(
    agent: NPCAgent,
    response_text: str | None = None,
    container=None,
) -> None:
    """Render one NPC card and its latest response."""
    target = container or st
    response_markup = (
        f'<div class="response-box">{response_text}</div>'
        if response_text
        else '<div class="response-box" style="color:#8d9aae">Waiting for event...</div>'
    )
    target.markdown(
        f"""
        <div class="agent-panel">
          <div class="status">● ONLINE / MEMORY ENABLED</div>
          <div class="agent-name">{agent.name}</div>
          <div class="agent-persona">{agent.persona}</div>
          {response_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_logs() -> None:
    """Display the event history captured in session state."""
    with st.expander("⌁ System Logs", expanded=bool(st.session_state.event_logs)):
        if not st.session_state.event_logs:
            st.caption("No events triggered yet.")
            return
        for log in reversed(st.session_state.event_logs):
            st.markdown(
                f'<div class="log-line">[{log["time"]}] EVENT → {log["event"]}</div>',
                unsafe_allow_html=True,
            )
            for response in log["responses"]:
                st.markdown(
                    f'<div class="log-line">[{log["time"]}] {response["name"]} → '
                    f'{response["text"]}</div>',
                    unsafe_allow_html=True,
                )


def main() -> None:
    load_dotenv()
    configure_page()

    st.markdown(
        """
        <div class="hero">
          <h1>⚙️ AI Multi-Agent Orchestrator Sandbox</h1>
          <p>Developer Dashboard · Observe NPCs react autonomously to events in a game world.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.error(
            "GEMINI_API_KEY was not found. Create a `.env` file next to app.py with "
            "`GEMINI_API_KEY=your-key`, then restart the app."
        )
        st.stop()

    if "agents" not in st.session_state or len(st.session_state.agents) != len(
        AGENT_DEFINITIONS
    ):
        try:
            st.session_state.agents = initialize_agents(api_key)
        except Exception as error:
            st.error(f"Could not initialize the NPCs: {error}")
            st.stop()
    if "event_logs" not in st.session_state:
        st.session_state.event_logs = []
    if "latest_responses" not in st.session_state:
        st.session_state.latest_responses = {}

    st.subheader("World Event")
    event = st.text_input(
        "Enter an event to orchestrate the NPCs",
        placeholder="Example: The granary at the North Gate is on fire!",
        label_visibility="collapsed",
    )
    trigger = st.button("⚡ Trigger Event", type="primary", use_container_width=True)

    columns = st.columns(1)
    panel_placeholders = [column.empty() for column in columns]
    for placeholder, agent in zip(panel_placeholders, st.session_state.agents):
        render_agent_panel(
            agent,
            st.session_state.latest_responses.get(agent.name),
            placeholder,
        )

    if trigger and not event.strip():
        st.warning("Enter a World Event before triggering the event.")
    elif trigger:
        for placeholder, agent in zip(panel_placeholders, st.session_state.agents):
            render_agent_panel(agent, FALLBACK_RESPONSE, placeholder)

        responses: dict[str, str] = {}
        for placeholder, agent in zip(panel_placeholders, st.session_state.agents):
            with st.spinner(f"{agent.name} is thinking..."):
                response = run_agents_sequential([agent], event.strip())[agent.name]
            if response.startswith(("Gemini ", "Error:")):
                response = FALLBACK_RESPONSE
            responses[agent.name] = response
            st.session_state.latest_responses[agent.name] = response
            render_agent_panel(agent, response, placeholder)

        st.session_state.event_logs.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "event": event.strip(),
                "responses": [
                    {
                        "name": agent.name,
                        "text": responses.get(agent.name, "No response"),
                    }
                    for agent in st.session_state.agents
                ],
            }
        )

    render_system_logs()


if __name__ == "__main__":
    main()

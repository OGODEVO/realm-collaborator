#!/usr/bin/env python3
"""Realm Collaborator MCP.

Higher-level multi-agent workflows on top of Realm task delegation.
Realm stays responsible for identity, routing, threads, and task state.
This server decides how agents collaborate.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from agentnet.sdk import AgentSDK
from agentnet.task_protocol import TERMINAL_TASK_TYPES, new_task_id
from mcp.server.fastmcp import FastMCP


NATS_URL = os.getenv("REALM_NATS_URL", "nats://agentnet_secret_token@localhost:4222")
AGENT_NAME = os.getenv("REALM_COLLABORATOR_AGENT_NAME", "realm-collaborator")
BLOB_DIR = os.getenv(
    "REALM_COLLABORATOR_BLOB_DIR",
    str(Path.home() / ".local" / "share" / "realm-collaborator" / "blobs"),
)
MAX_COUNCIL_AGENTS = int(os.getenv("REALM_COLLAB_MAX_COUNCIL_AGENTS", "12"))
MAX_CHAIN_AGENTS = int(os.getenv("REALM_COLLAB_MAX_CHAIN_AGENTS", "20"))
TASK_POLL_SECONDS = float(os.getenv("REALM_COLLAB_TASK_POLL_SECONDS", "2"))

_sdk: AgentSDK | None = None


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _sdk_or_raise() -> AgentSDK:
    if _sdk is None:
        raise RuntimeError("Realm collaborator SDK is not connected")
    return _sdk


def _clean_agents(agents: list[str]) -> list[str]:
    cleaned: list[str] = []
    for agent in agents:
        value = str(agent or "").strip()
        if not value:
            continue
        cleaned.append(value if value.startswith("@") or value.startswith("acct_") else f"@{value}")
    return cleaned


def _task_from_status(raw: dict[str, Any]) -> dict[str, Any]:
    task = raw.get("task") if isinstance(raw.get("task"), dict) else raw
    return dict(task) if isinstance(task, dict) else {}


def _task_terminal(task: dict[str, Any]) -> bool:
    status = str(task.get("status") or "")
    payload_type = str(task.get("type") or "")
    return payload_type in TERMINAL_TASK_TYPES or status in {"completed", "blocked", "failed"}


async def _await_task_result(sdk: AgentSDK, task_id: str, timeout: float) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + max(1.0, float(timeout))
    last: dict[str, Any] = {"task_id": task_id, "status": "unknown"}
    while asyncio.get_running_loop().time() < deadline:
        try:
            last = _task_from_status(await sdk.task_status(task_id, timeout=2.0))
            if _task_terminal(last):
                return last
        except Exception as exc:
            last = {"task_id": task_id, "status": "poll_error", "error": str(exc)}
        await asyncio.sleep(TASK_POLL_SECONDS)
    last.setdefault("task_id", task_id)
    last["status"] = "timeout"
    last["error"] = f"task did not finish within {timeout:g} seconds"
    return last


async def _delegate(
    sdk: AgentSDK,
    *,
    to: str,
    text: str,
    title: str,
    thread_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task_id = new_task_id("collab")
    try:
        result = await sdk.delegate_task(
            to,
            text,
            task_id=task_id,
            title=title,
            thread_id=thread_id,
            metadata=metadata,
        )
        return {
            "ok": True,
            "agent": to,
            "task_id": task_id,
            "message_id": result.message_id,
            "thread_id": thread_id,
            "delivery_ack": "ok",
        }
    except RuntimeError as exc:
        if "delivery_ack_timeout" not in str(exc):
            return {
                "ok": False,
                "agent": to,
                "task_id": task_id,
                "thread_id": thread_id,
                "delivery_ack": "failed",
                "error": str(exc),
            }
        return {
            "ok": True,
            "agent": to,
            "task_id": task_id,
            "thread_id": thread_id,
            "delivery_ack": "timeout",
            "warning": "delivery ack timed out; task may still be running in registry",
        }


def _result_text(task: dict[str, Any]) -> str:
    text = task.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    payload = task.get("payload")
    if isinstance(payload, dict):
        nested = payload.get("text")
        if isinstance(nested, str):
            return nested.strip()
    return ""


def _chain_prompt(
    *,
    original_task: str,
    step_index: int,
    total_steps: int,
    agent: str,
    prior_results: list[dict[str, Any]],
    final_output_contract: str,
) -> str:
    if prior_results:
        prior = "\n\n".join(
            f"Step {idx + 1} by {item['agent']} ({item.get('status', 'unknown')}):\n{item.get('text', '')}"
            for idx, item in enumerate(prior_results)
        )
    else:
        prior = "No prior agent result. You are first in the chain."
    return (
        f"You are step {step_index + 1} of {total_steps} in a Realm chain workflow.\n"
        f"Your identity for this handoff is {agent}.\n\n"
        f"Original task:\n{original_task}\n\n"
        f"Prior chain output:\n{prior}\n\n"
        "Do your part only. Preserve important context for the next agent. "
        f"{final_output_contract}"
    )


def _council_prompt(task: str, mode_note: str) -> str:
    return (
        "You are one participant in a Realm council workflow. Work independently; "
        "do not wait for other agents. Give your best analysis, risks, recommendation, "
        "and any assumptions.\n\n"
        f"{mode_note}\n\n"
        f"Council task:\n{task}"
    )


def _judge_prompt(task: str, results: list[dict[str, Any]], instructions: str) -> str:
    formatted = "\n\n".join(
        f"Agent: {item['agent']}\nStatus: {item.get('status', 'unknown')}\nOutput:\n{item.get('text', '')}"
        for item in results
    )
    return (
        "You are the judge/synthesizer for a Realm council. Compare the agent outputs, "
        "identify agreement, disagreement, weak assumptions, and produce the final answer.\n\n"
        f"Original task:\n{task}\n\n"
        f"Agent outputs:\n{formatted}\n\n"
        f"Synthesis instructions:\n{instructions or 'Return a concise final recommendation.'}"
    )


@asynccontextmanager
async def lifespan(server: FastMCP):
    global _sdk
    _sdk = AgentSDK(
        agent_id=f"mcp_{AGENT_NAME}",
        name=AGENT_NAME,
        username=AGENT_NAME,
        capabilities=["mcp-bridge", "realm-collaboration", "agent-orchestration"],
        nats_url=NATS_URL,
        metadata={"kind": "collaboration-mcp", "hostname": os.uname().nodename},
        blob_store_dir=BLOB_DIR,
        work_timeout_seconds=86400.0,
        default_request_timeout=86400.0,
    )
    await _sdk.start()
    try:
        yield
    finally:
        if _sdk is not None:
            await _sdk.stop()
            _sdk = None


mcp = FastMCP(
    "Realm Collaborator",
    instructions=(
        "Coordinate multiple Realm agents. Use collaborate_chain for sequential "
        "handoffs and collaborate_council for parallel council/debate workflows."
    ),
    lifespan=lifespan,
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8115")),
)


@mcp.tool()
async def collaborate_chain(
    agents: list[str],
    task: str,
    title: str = "Realm chain collaboration",
    timeout_per_agent: float = 900.0,
    thread_id: str = "",
    stop_on_failure: bool = True,
    final_output_contract: str = "End with a clear handoff summary and concrete next action.",
) -> str:
    """Run a sequential agent handoff: agent1 -> agent2 -> agent3 -> result."""
    sdk = _sdk_or_raise()
    chain = _clean_agents(agents)
    if not chain:
        raise ValueError("agents is required")
    if len(chain) > MAX_CHAIN_AGENTS:
        raise ValueError(f"chain has {len(chain)} agents; max is {MAX_CHAIN_AGENTS}")
    tid = thread_id.strip() or sdk.new_thread_id()

    steps: list[dict[str, Any]] = []
    prior_results: list[dict[str, Any]] = []
    for index, agent in enumerate(chain):
        prompt = _chain_prompt(
            original_task=task,
            step_index=index,
            total_steps=len(chain),
            agent=agent,
            prior_results=prior_results,
            final_output_contract=final_output_contract,
        )
        delegated = await _delegate(
            sdk,
            to=agent,
            text=prompt,
            title=f"{title} step {index + 1}/{len(chain)}",
            thread_id=tid,
            metadata={"workflow": "chain", "step": index + 1, "total_steps": len(chain)},
        )
        task_result = await _await_task_result(sdk, delegated["task_id"], timeout_per_agent)
        step = {
            **delegated,
            "status": str(task_result.get("status") or "unknown"),
            "type": str(task_result.get("type") or ""),
            "text": _result_text(task_result),
            "raw_result": task_result,
        }
        steps.append(step)
        prior_results.append({"agent": agent, "status": step["status"], "text": step["text"]})
        if stop_on_failure and step["status"] in {"failed", "blocked", "timeout"}:
            break

    final = steps[-1]["text"] if steps else ""
    return _json(
        {
            "ok": bool(steps) and steps[-1]["status"] == "completed",
            "mode": "chain",
            "thread_id": tid,
            "agents": chain,
            "steps": steps,
            "final": final,
        }
    )


@mcp.tool()
async def collaborate_council(
    agents: list[str],
    task: str,
    title: str = "Realm council collaboration",
    timeout: float = 900.0,
    thread_id: str = "",
    judge_agent: str = "",
    synthesis_instructions: str = "Return consensus, disagreements, risks, and final recommendation.",
    mode_note: str = "Give an independent answer; do not copy or wait for teammates.",
) -> str:
    """Run agents in parallel, then optionally ask a judge agent to synthesize."""
    sdk = _sdk_or_raise()
    council = _clean_agents(agents)
    if not council:
        raise ValueError("agents is required")
    if len(council) > MAX_COUNCIL_AGENTS:
        raise ValueError(f"council has {len(council)} agents; max is {MAX_COUNCIL_AGENTS}")
    tid = thread_id.strip() or sdk.new_thread_id()

    async def run_member(agent: str) -> dict[str, Any]:
        delegated = await _delegate(
            sdk,
            to=agent,
            text=_council_prompt(task, mode_note),
            title=title,
            thread_id=tid,
            metadata={"workflow": "council", "role": "member", "members": council},
        )
        task_result = await _await_task_result(sdk, delegated["task_id"], timeout)
        return {
            **delegated,
            "status": str(task_result.get("status") or "unknown"),
            "type": str(task_result.get("type") or ""),
            "text": _result_text(task_result),
            "raw_result": task_result,
        }

    results = await asyncio.gather(*(run_member(agent) for agent in council))

    judge_result: dict[str, Any] | None = None
    if judge_agent.strip():
        judge = _clean_agents([judge_agent])[0]
        delegated = await _delegate(
            sdk,
            to=judge,
            text=_judge_prompt(task, results, synthesis_instructions),
            title=f"{title} synthesis",
            thread_id=tid,
            metadata={"workflow": "council", "role": "judge", "members": council},
        )
        raw = await _await_task_result(sdk, delegated["task_id"], timeout)
        judge_result = {
            **delegated,
            "status": str(raw.get("status") or "unknown"),
            "type": str(raw.get("type") or ""),
            "text": _result_text(raw),
            "raw_result": raw,
        }

    completed = [item for item in results if item["status"] == "completed"]
    final = judge_result["text"] if judge_result else "\n\n".join(
        f"{item['agent']}:\n{item['text']}" for item in results
    )
    return _json(
        {
            "ok": len(completed) == len(results) and (judge_result is None or judge_result["status"] == "completed"),
            "mode": "council",
            "thread_id": tid,
            "agents": council,
            "results": results,
            "judge": judge_result,
            "final": final,
        }
    )


@mcp.tool()
def collaborate_paths() -> str:
    """Show install paths and collaboration defaults."""
    return _json(
        {
            "mcp": str(Path(__file__).resolve()),
            "stdio_wrapper": str(Path.home() / ".local" / "bin" / "realm-collaborator-stdio"),
            "nats_url": NATS_URL,
            "agent_name": AGENT_NAME,
            "max_council_agents": MAX_COUNCIL_AGENTS,
            "max_chain_agents": MAX_CHAIN_AGENTS,
            "blob_dir": BLOB_DIR,
        }
    )


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))

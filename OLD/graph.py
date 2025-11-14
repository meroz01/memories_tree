from __future__ import annotations

import datetime
import json
import os
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from model import build_llm

# ============================================================
# CONFIG
# ============================================================

MEMORY_FILE = "memories.json"

# Promote short -> mid after 1 hour
SHORT_MAX_AGE_HOURS = 1.0

# Promote mid -> long after 7 days
MID_TO_LONG_DAYS = 7.0


# ============================================================
# TYPES
# ============================================================

MemoryType = Literal["description", "task", "goal", "personal_info", "other"]
MemoryLayer = Literal["short", "mid", "long"]


class MemoryItem(TypedDict):
    type: MemoryType
    layer: MemoryLayer
    content: str
    source_message_index: int
    strength: float          # "importance"/salience score
    created_at: int          # UNIX timestamp
    updated_at: int          # UNIX timestamp
    tags: list[str]


class ConversationState(TypedDict):
    messages: list[BaseMessage]
    short_memories: list[MemoryItem]
    mid_memories: list[MemoryItem]
    long_memories: list[MemoryItem]


llm = build_llm()


# ============================================================
# TIME HELPERS
# ============================================================

def _now_ts() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp())


# ============================================================
# FILE STORAGE
# ============================================================

def load_memories_from_file() -> tuple[list[MemoryItem], list[MemoryItem], list[MemoryItem]]:
    if not os.path.exists(MEMORY_FILE):
        print("[memory_store] No memory file found. Starting fresh.")
        return [], [], []

    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[memory_store] Failed to load {MEMORY_FILE}: {e}")
        return [], [], []

    short = data.get("short_memories", []) or []
    mid = data.get("mid_memories", []) or []
    long_ = data.get("long_memories", []) or []

    print(
        f"[memory_store] Loaded: short={len(short)}, mid={len(mid)}, long={len(long_)}"
    )

    short, mid, long_ = reorganize_memories_on_load(short, mid, long_)

    return short, mid, long_


def save_memories_to_file(state: ConversationState) -> None:
    payload = {
        "short_memories": state["short_memories"],
        "mid_memories": state["mid_memories"],
        "long_memories": state["long_memories"],
    }

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[memory_store] Failed to save {MEMORY_FILE}: {e}")
        return

    print(
        f"[memory_store] Saved memories: "
        f"short={len(state['short_memories'])}, "
        f"mid={len(state['mid_memories'])}, "
        f"long={len(state['long_memories'])}"
    )


# ============================================================
# MEMORY REORGANIZATION (decay + promotion on load)
# ============================================================

def reorganize_memories_on_load(
    short: list[MemoryItem],
    mid: list[MemoryItem],
    long_: list[MemoryItem],
) -> tuple[list[MemoryItem], list[MemoryItem], list[MemoryItem]]:

    now_ts = _now_ts()

    new_short: list[MemoryItem] = []
    new_mid: list[MemoryItem] = []
    new_long: list[MemoryItem] = []

    promoted_short = 0
    promoted_mid = 0

    # ---- SHORT → MID ----
    for mem in short:
        # decay strength by 1
        mem["strength"] = max(0.0, mem.get("strength", 0.0) - 1.0)

        created_ts = mem.get("created_at", now_ts)
        age_hours = (now_ts - created_ts) / 3600.0

        if age_hours > SHORT_MAX_AGE_HOURS:
            mem["layer"] = "mid"
            mem["updated_at"] = now_ts
            new_mid.append(mem)
            promoted_short += 1
        else:
            new_short.append(mem)

    # ---- MID → LONG ----
    for mem in mid:
        mem["strength"] = max(0.0, mem.get("strength", 0.0) - 1.0)

        created_ts = mem.get("created_at", now_ts)
        age_days = (now_ts - created_ts) / 86400.0

        if age_days > MID_TO_LONG_DAYS:
            mem["layer"] = "long"
            mem["updated_at"] = now_ts
            new_long.append(mem)
            promoted_mid += 1
        else:
            new_mid.append(mem)

    # ---- LONG stays LONG (but decays strength) ----
    for mem in long_:
        mem["strength"] = max(0.0, mem.get("strength", 0.0) - 1.0)
        new_long.append(mem)

    print(
        f"[memory_reorg] Promotions: short→mid={promoted_short}, mid→long={promoted_mid}. "
        f"Now short={len(new_short)}, mid={len(new_mid)}, long={len(new_long)}"
    )

    return new_short, new_mid, new_long


# ============================================================
# MEMORY RETRIEVAL (with +10 strength on recall)
# ============================================================

def _get_last_user_message(state: ConversationState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _score_memory(mem: MemoryItem, query: str, total_messages: int) -> float:
    query_tokens = set(query.lower().split())
    content_tokens = set(mem["content"].lower().split())

    overlap = (
        len(query_tokens & content_tokens) / len(query_tokens)
        if query_tokens else 0.0
    )

    recency = 1.0 - (
        (total_messages - 1 - mem["source_message_index"]) / max(total_messages, 1)
    )

    layer_weights = {"short": 0.7, "mid": 1.0, "long": 0.9}
    layer_weight = layer_weights.get(mem["layer"], 0.9)

    strength = mem.get("strength", 1.0)

    return strength * layer_weight * (0.4 + 0.4 * overlap + 0.2 * recency)


def get_relevant_memories(
    state: ConversationState, max_items: int = 8, min_score: float = 0.25
) -> list[tuple[MemoryItem, float]]:
    query = _get_last_user_message(state)
    if not query:
        return []

    all_mems = (
        state["short_memories"]
        + state["mid_memories"]
        + state["long_memories"]
    )

    total_messages = len(state["messages"])

    scored: list[tuple[MemoryItem, float]] = []
    for mem in all_mems:
        score = _score_memory(mem, query, total_messages)
        if score >= min_score:
            scored.append((mem, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:max_items]

    # 🔼 boost strength for recalled memories
    if selected:
        now_ts = _now_ts()
        for mem, _ in selected:
            mem["strength"] = mem.get("strength", 0.0) + 10.0
            mem["updated_at"] = now_ts

        # we don't save here, save will happen in memory_node,
        # but if you want immediate persistence uncomment:
        # save_memories_to_file(state)

    return selected


def get_always_on_short_memories(state: ConversationState, max_items: int = 3):
    return state["short_memories"][-max_items:]


# ============================================================
# MEMORY EXTRACTION NODE
# ============================================================

def memory_node(state: ConversationState) -> ConversationState:
    messages = state["messages"]

    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        print("[memory_node] No HumanMessage found, skipping extraction.")
        return state

    last_user_content = messages[last_human_idx].content
    print(f"[memory_node] Extracting memories from message index={last_human_idx}: {last_user_content!r}")

    system = SystemMessage(
        content=(
            "You are a memory extraction module.\n"
            "You receive a single user message.\n\n"
            "Task:\n"
            "- Split it into small, atomic pieces (facts, tasks, goals, personal info).\n"
            "- For each piece, set:\n"
            "  - type: one of [description, task, goal, personal_info, other]\n"
            "  - layer: one of [short, mid, long]\n"
            "    * short: very transient (one-off question, immediate context)\n"
            "    * mid: ongoing tasks/goals, recent facts (days/weeks)\n"
            "    * long: stable personal info, long-term goals, preferences\n\n"
            "Return ONLY a JSON object, with no markdown and no extra text:\n"
            "{ \"memories\": [ { \"type\": \"task\", \"layer\": \"mid\", \"content\": \"...\", \"tags\": [\"tag1\"] }, ... ] }\n"
        )
    )

    extraction = llm.invoke([system, HumanMessage(content=last_user_content)])

    if not isinstance(extraction.content, str):
        print("[memory_node] Non-string extraction.content, skipping.")
        return state

    print(f"[memory_node] Raw extraction from LLM: {extraction.content!r}")

    try:
        data = json.loads(extraction.content)
        items = data.get("memories", [])
    except Exception as e:
        print(f"[memory_node] JSON parse error: {e}")
        return state

    new_short: list[MemoryItem] = list(state["short_memories"])
    new_mid: list[MemoryItem] = list(state["mid_memories"])
    new_long: list[MemoryItem] = list(state["long_memories"])

    added_counts = {"short": 0, "mid": 0, "long": 0}
    added_chars = {"short": 0, "mid": 0, "long": 0}

    now_ts = _now_ts()

    for item in items:
        content = (item.get("content") or "").strip()
        if not content:
            continue

        layer_raw = item.get("layer", "mid")
        layer: MemoryLayer = layer_raw if layer_raw in ("short", "mid", "long") else "mid"

        mem: MemoryItem = {
            "type": item.get("type", "other"),
            "layer": layer,
            "content": content,
            "source_message_index": last_human_idx,
            "strength": 1.0,          # starting strength
            "created_at": now_ts,
            "updated_at": now_ts,
            "tags": item.get("tags", []),
        }

        size = len(content)

        if layer == "short":
            new_short.append(mem)
        elif layer == "mid":
            new_mid.append(mem)
        else:
            new_long.append(mem)

        added_counts[layer] += 1
        added_chars[layer] += size

    state["short_memories"] = new_short
    state["mid_memories"] = new_mid
    state["long_memories"] = new_long

    print(
        f"[memory_node] Added memories: "
        f"short={added_counts['short']} ({added_chars['short']} chars), "
        f"mid={added_counts['mid']} ({added_chars['mid']} chars), "
        f"long={added_counts['long']} ({added_chars['long']} chars). "
        f"Totals now: short={len(new_short)}, mid={len(new_mid)}, long={len(new_long)}"
    )

    save_memories_to_file(state)

    return state


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ConversationState) -> ConversationState:
    base_messages = state["messages"]

    short_mems = get_always_on_short_memories(state)
    relevant = get_relevant_memories(state)

    memory_lines: list[str] = []

    if short_mems:
        memory_lines.append("Short-term working memory:")
        for mem in short_mems:
            memory_lines.append(f"- ({mem['type']}) {mem['content']}")

    if relevant:
        memory_lines.append("\nRelevant memories:")
        for mem, score in relevant:
            memory_lines.append(f"- ({mem['layer']}/{mem['type']}) {mem['content']}")

    memory_context = "\n".join(memory_lines)

    print(
        f"[chat_node] Injecting memory context: short={len(short_mems)}, relevant={len(relevant)}"
    )

    messages_for_llm: list[BaseMessage] = []

    if memory_context:
        messages_for_llm.append(
            SystemMessage(
                content=(
                    "Memory context (do not repeat verbatim; just use it as knowledge):\n\n"
                    + memory_context
                )
            )
        )

    messages_for_llm.extend(base_messages)

    response = llm.invoke(messages_for_llm)

    state["messages"] = base_messages + [response]
    return state


# ============================================================
# GRAPH WIRING
# ============================================================

def build_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("memory", memory_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "chat")
    graph.add_edge("chat", END)

    return graph.compile()
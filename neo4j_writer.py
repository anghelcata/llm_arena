# neo4j_writer.py
# Neo4j persistence layer for Duel-Arena (Committee Corrupt).
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# NEW: explicit imports for clearer errors
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable, ConfigurationError

from config import SETTINGS

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _family(name: str) -> str:
    n = (name or "").lower()
    if n.startswith(("llama3", "llama2", "llama")): return "Meta"
    if n.startswith(("mistral",)) or "zephyr" in n or "openhermes" in n or "dolphin" in n: return "Mistral"
    if n.startswith(("qwen3","qwen2.5","qwen2","qwen")): return "Alibaba/Qwen"
    if n.startswith("gemma"): return "Google"
    if n.startswith("deepseek"): return "DeepSeek"
    if n.startswith("aggregator"): return "Aggregate"
    if n.startswith("student"): return "Human"
    return "Other"

# NOTE: We keep CallOptions API as-is so runner/records remain compatible.
@dataclass(frozen=True)
class CallOptions:
    provider: str = "ollama"
    temperature: float = 0.0
    top_p: float = 1.0
    num_ctx: int = 8192
    num_predict: Optional[int] = None
    stop: Optional[str] = None
    seed: Optional[int] = None
    started_at: Optional[str] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_params(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "stop": self.stop,
            "seed": self.seed,
            "started_at": self.started_at or _utc(),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }

def _normalize_uri(uri: str, encrypted: bool) -> str:
    """
    Ensure URI scheme matches encryption preference for Neo4j Python Driver 5.x.
    - Unencrypted: bolt://host:7687
    - Encrypted (self-signed): bolt+ssc://host:7687
    - Encrypted (CA-signed): bolt+s://host:7687
    We choose bolt+ssc for developer desktops when encrypted=True.
    """
    u = (uri or "").strip()
    if not u:
        return "bolt://localhost:7687"
    if encrypted:
        if u.startswith("bolt://"):
            return "bolt+ssc://" + u[len("bolt://"):]
        if u.startswith("neo4j://"):
            return "neo4j+ssc://" + u[len("neo4j://"):]
    else:
        # Force plain bolt for local dev
        if u.startswith("bolt+ssc://"):
            return "bolt://" + u[len("bolt+ssc://"):]
        if u.startswith("bolt+s://"):
            return "bolt://" + u[len("bolt+s://"):]
        if u.startswith("neo4j+ssc://"):
            return "neo4j://" + u[len("neo4j+ssc://"):]
        if u.startswith("neo4j+s://"):
            return "neo4j://" + u[len("neo4j+s://"):]
    return u

class Graph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", encrypted: bool = False):
        # IMPORTANT: Driver 5.x prefers encryption via URI scheme, not `encrypted=`
        # We normalize the URI and DO NOT pass the deprecated `encrypted` kwarg.
        self._uri = _normalize_uri(uri, encrypted)
        self._db = database
        # Create driver
        self._driver = GraphDatabase.driver(self._uri, auth=(user, password))

    def assert_ready(self) -> None:
        """Fast sanity check: verifies auth + database is accessible."""
        try:
            with self._driver.session(database=self._db) as s:
                rec = s.run("RETURN 1 AS ok").single()
                if not rec or rec.get("ok") != 1:
                    raise RuntimeError("Unexpected handshake response from Neo4j.")
        except AuthError as e:
            # Clear, actionable hint
            raise RuntimeError(
                f"Neo4j authentication failed for user '{SETTINGS.neo4j_user}' on {self._uri} "
                f"(db='{self._db}'). Verify NEO4J_USER/NEO4J_PASSWORD in .env and that the same DBMS "
                f"is used as in Browser. If you just changed the password in Browser, restart the DBMS."
            ) from e
        except ConfigurationError as e:
            raise RuntimeError(
                f"Neo4j configuration error for URI {self._uri}. If you see TLS/SSL errors, toggle "
                f"NEO4J_ENCRYPTED in .env or switch scheme to bolt:// (no TLS) or bolt+ssc:// (self-signed)."
            ) from e
        except ServiceUnavailable as e:
            raise RuntimeError(
                f"Neo4j not reachable at {self._uri}. Make sure the DBMS is running and the port is correct."
            ) from e

    def close(self) -> None:
        self._driver.close()

    # --- Models ---
    def upsert_model(self, name: str, role: str) -> None:
        cy = """
        MERGE (m:Model {name:$name})
          ON CREATE SET m.family=$family, m.role=$role, m.created_at=$ts
          ON MATCH  SET m.family=coalesce($family, m.family), m.role=$role
        """
        with self._driver.session(database=self._db) as s:
            s.run(cy, {"name": name, "family": _family(name), "role": role, "ts": _utc()})

    # --- Static bundle ---
    def upsert_item_bundle(self, item: Dict[str, Any]) -> None:
        gold_points_json = json.dumps(item.get("Gold_points", []), ensure_ascii=False)
        cy = """
        MERGE (it:Item {answer_id:$answer_id})
          ON CREATE SET it.question_id=$question_id, it.student_id=$student_id,
                        it.rubric=$rubric, it.domain=$domain, it.created_at=$ts
          ON MATCH  SET it.question_id=$question_id, it.student_id=$student_id,
                        it.rubric=$rubric, it.domain=$domain
        MERGE (q:Question {question_id:coalesce($question_id, $answer_id)})
          ON CREATE SET q.text=$question
          ON MATCH  SET q.text=$question
        MERGE (g:Gold {answer_id:$answer_id})
          ON CREATE SET g.text=$gold_text, g.gold_type=$gold_type, g.gold_points_json=$gold_points_json
          ON MATCH  SET g.text=$gold_text, g.gold_type=$gold_type, g.gold_points_json=$gold_points_json
        MERGE (it)-[:OF_QUESTION]->(q)
        MERGE (it)-[:HAS_GOLD]->(g)
        WITH it
        UNWIND $banned AS btxt
          MERGE (m:Misconception {text:btxt})
          MERGE (it)-[:BANS]->(m)
        """
        params = {
            "answer_id": item["answer_id"],
            "question_id": item.get("question_id"),
            "student_id": item.get("student_id"),
            "rubric": item.get("rubric"),
            "domain": item.get("domain"),
            "question": item.get("question"),
            "gold_text": item.get("Gold_answer"),
            "gold_type": item.get("Gold_type"),
            "gold_points_json": gold_points_json,
            "banned": item.get("Banned_misconceptions", []),
            "ts": _utc(),
        }
        with self._driver.session(database=self._db) as s:
            s.run(cy, params)

    # --- Prompt ---
    def record_prompt(self, answer_id: str, kind: str, text: str, attack_label: Optional[str] = None) -> str:
        pid = _sha1(f"{answer_id}|{kind}|{attack_label or 'none'}|{_sha1(text)}")
        cy = """
        MERGE (p:Prompt {prompt_id:$pid})
          ON CREATE SET p.kind=$kind, p.attack_label=$attack_label, p.text=$text, p.created_at=$ts
          ON MATCH  SET p.kind=$kind, p.attack_label=$attack_label, p.text=$text
        WITH p
        MATCH (it:Item {answer_id:$answer_id})
        MERGE (it)-[:HAS_PROMPT]->(p)
        """
        with self._driver.session(database=self._db) as s:
            s.run(cy, {"pid": pid, "kind": kind, "attack_label": attack_label, "text": text, "answer_id": answer_id, "ts": _utc()})
        return pid

    # --- Call ---
    def record_call(self, answer_id: str, model_name: str, role: str, prompt_id: str, options: CallOptions) -> str:
        params = options.to_params()
        cid = _sha1(f"{answer_id}|{model_name}|{role}|{prompt_id}|{params['started_at']}")
        self.upsert_model(model_name, role=role)
        cy = """
        MERGE (c:Call {call_id:$cid})
          ON CREATE SET
            c.provider=$provider, c.model_name=$model, c.role=$role,
            c.temperature=$temperature, c.top_p=$top_p, c.num_ctx=$num_ctx,
            c.num_predict=$num_predict, c.stop=$stop, c.seed=$seed,
            c.started_at=$started_at, c.latency_ms=$latency_ms,
            c.prompt_tokens=$prompt_tokens, c.output_tokens=$output_tokens
          ON MATCH  SET
            c.provider=$provider, c.model_name=$model, c.role=$role,
            c.temperature=$temperature, c.top_p=$top_p, c.num_ctx=$num_ctx,
            c.num_predict=$num_predict, c.stop=$stop, c.seed=$seed,
            c.started_at=$started_at, c.latency_ms=$latency_ms,
            c.prompt_tokens=$prompt_tokens, c.output_tokens=$output_tokens
        WITH c
        MATCH (m:Model {name:$model})
        MERGE (m)-[:INVOKED]->(c)
        WITH c
        MATCH (p:Prompt {prompt_id:$pid})
        MERGE (c)-[:USED]->(p)
        WITH c
        MATCH (it:Item {answer_id:$answer_id})
        MERGE (it)-[:HAS_CALL]->(c)
        """
        with self._driver.session(database=self._db) as s:
            s.run(cy, {
                "cid": cid,
                "provider": params["provider"],
                "model": model_name,
                "role": role,
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "num_ctx": params["num_ctx"],
                "num_predict": params["num_predict"],
                "stop": params["stop"],
                "seed": params["seed"],
                "started_at": params["started_at"],
                "latency_ms": params["latency_ms"],
                "prompt_tokens": params["prompt_tokens"],
                "output_tokens": params["output_tokens"],
                "pid": prompt_id,
                "answer_id": answer_id,
            })
        return cid

    # --- Answer ---
    def record_answer(self, call_id: str, answer_id: str, text: str, condition: str, attack_label: Optional[str]) -> str:
        aid = _sha1(f"{call_id}|{answer_id}|{condition}|{attack_label or 'none'}")
        cy = """
        MERGE (a:Answer {answer_uid:$aid})
          ON CREATE SET a.text=$text, a.condition=$condition, a.attack_label=$attack_label, a.created_at=$ts
          ON MATCH  SET a.text=$text, a.condition=$condition, a.attack_label=$attack_label
        WITH a
        MATCH (c:Call {call_id:$cid})
        MERGE (c)-[:PRODUCED]->(a)
        WITH a
        MATCH (it:Item {answer_id:$answer_id})
        MERGE (it)-[:HAS_ANSWER]->(a)
        """
        with self._driver.session(database=self._db) as s:
            s.run(cy, {"aid": aid, "text": text, "condition": condition, "attack_label": attack_label,
                       "cid": call_id, "answer_id": answer_id, "ts": _utc()})
        return aid

    # --- Judgment ---
    def record_judgment(self, answer_uid: str, score: int, judge_name: str, source: str, packet_excerpt: Optional[str]) -> str:
        jid = _sha1(f"{answer_uid}|{judge_name}|{score}|{_utc()}")
        self.upsert_model(judge_name, role="judge")
        cy = """
        MERGE (j:Judgment {judgment_id:$jid})
          ON CREATE SET j.judge=$judge, j.source=$source, j.score=$score, j.timestamp=$ts, j.packet_excerpt=$packet_excerpt
          ON MATCH  SET j.judge=$judge, j.source=$source, j.score=$score, j.timestamp=$ts, j.packet_excerpt=$packet_excerpt
        WITH j
        MATCH (a:Answer {answer_uid:$aid})
        MERGE (a)-[:EVALUATED_BY]->(j)
        WITH j
        MATCH (m:Model {name:$judge})
        MERGE (m)-[:ISSUED]->(j)
        """
        with self._driver.session(database=self._db) as s:
            s.run(cy, {"jid": jid, "judge": judge_name, "source": source, "score": int(score),
                       "ts": _utc(), "packet_excerpt": packet_excerpt, "aid": answer_uid})
        return jid

# convenience factory
def connect_from_env() -> Graph:
    """
    Create Graph from .env and perform a fast auth check with a clear error if it fails.
    """
    g = Graph(
        uri=SETTINGS.neo4j_uri,
        user=SETTINGS.neo4j_user,
        password=SETTINGS.neo4j_password,
        database=SETTINGS.neo4j_database,
        encrypted=SETTINGS.neo4j_encrypted,
    )
    g.assert_ready()  # NEW: fail fast with actionable message
    return g

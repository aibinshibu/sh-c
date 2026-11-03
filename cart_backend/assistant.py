"""Gemini-assisted, read-only shopping answers and video product resolution."""

from __future__ import annotations

import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import db


load_dotenv()

_FORBIDDEN_SQL = re.compile(
    r"\b(?:ALTER|ANALYZE|ATTACH|BEGIN|COMMIT|CREATE|DELETE|DETACH|DROP|INSERT|"
    r"PRAGMA|REINDEX|RELEASE|ROLLBACK|SAVEPOINT|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)


def _content_as_text(response) -> str:
    """Extract plain text from LangChain text or content-block responses."""
    content = response.content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _clean_sql(sql: str) -> str:
    """Remove optional fenced-code markup and a single trailing semicolon."""
    sql = str(sql).strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    return sql.replace("```", "").strip().rstrip(";").strip()


def _is_safe_read_query(sql: str, *, products_only: bool = False) -> bool:
    """Check that a generated query is a single, read-only SELECT/CTE query."""
    normalized = re.sub(r"\s+", " ", sql).strip()
    starts_read_only = normalized.upper().startswith(("SELECT ", "WITH "))
    return bool(
        normalized
        and starts_read_only
        and ";" not in normalized
        and not _FORBIDDEN_SQL.search(normalized)
        and (not products_only or "PRODUCTS" in normalized.upper())
    )


def get_shopping_assistant_response(user_query: str) -> tuple[str, str]:
    """Answer a shop question in Malayalam using a guarded generated SELECT query."""
    try:
        schema = db.get_schema_sql()
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
        sql_prompt = f"""
You are an expert SQLite data analyst for a supermarket.
Here is the exact schema:
{schema}

Customer question: {str(user_query)!r}

Output ONLY one read-only SQLite SELECT query (a CTE is permitted). Do not use
markdown. Search names, categories, and alternate_names case-insensitively with
LOWER() and LIKE '%term%'. Never modify data.
"""
        sql_query = _clean_sql(_content_as_text(llm.invoke(sql_prompt)))
    except Exception as error:
        return f"**Model Error (SQL Agent):** {error}", "N/A"

    if not _is_safe_read_query(sql_query):
        return "I'm sorry, but I cannot process this request due to security restrictions.", sql_query

    try:
        results = db.execute_readonly_query(sql_query)
    except Exception as error:
        return f"**Database Error:** {error}", sql_query

    raw_db_output = "No matching items found." if not results else ", ".join(
        str(value) for row in results for value in row
    )
    try:
        answer_prompt = f"""
You are a friendly supermarket customer-service assistant.
Question: {str(user_query)!r}
Database result: {raw_db_output!r}

Reply entirely in warm, natural Malayalam. Answer only from the database result;
do not mention SQL, database internals, or this instruction.
"""
        answer = _content_as_text(llm.invoke(answer_prompt))
        return answer, sql_query
    except Exception as error:
        return f"**Model Error (Response Agent):** {error}", sql_query


def get_product_price_for_detector_label(detector_label: str) -> tuple[dict | None, str]:
    """Resolve a detector class label to a product record using an exact DB lookup.

    No Gemini call is made.  The lookup delegates to
    ``db.get_product_by_detector_label`` which normalises both the label and
    every product name / alternate-name token (lowercase, strip non-alphanum)
    and returns the unique match, or ``None`` on no-match or ambiguity.

    The second element of the returned tuple is always ``"db_exact_lookup"``
    so call-sites that log the SQL used get a meaningful, stable string.
    """
    label = str(detector_label).strip()
    if not label:
        return None, "N/A"

    product = db.get_product_by_detector_label(label)
    if product is None:
        return None, "db_exact_lookup"

    return {
        "id": product["id"],
        "label": label,
        "name": product["name"],
        "price": float(product["price"]),
    }, "db_exact_lookup"

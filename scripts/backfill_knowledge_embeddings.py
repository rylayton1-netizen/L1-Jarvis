# scripts/backfill_knowledge_embeddings.py
import os
import sys
import time
import argparse
from typing import List, Sequence

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Importing registers pgvector adapters so Python lists bind to vector columns
from pgvector.sqlalchemy import Vector  # noqa: F401

DEFAULT_MODEL = "text-embedding-3-small"  # 1536 dims (HNSW-friendly)
MAX_INPUT_LEN = 8000
DEFAULT_BATCH = 64

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill pgvector embeddings for public.knowledge")
    p.add_argument("--database-url", help="Override DATABASE_URL (postgres DSN)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI embedding model")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    p.add_argument("--company-id", type=int, help="Optional company_id filter")
    p.add_argument("--dry-run", action="store_true", help="Compute embeddings but do not write to DB")
    return p.parse_args()

def get_env(args: argparse.Namespace):
    load_dotenv()
    db_url = args.database_url or os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    api_key = os.getenv("OPENAI_API_KEY")
    if not db_url:
        print("ERROR: DATABASE_URL (or SQLALCHEMY_DATABASE_URI) is required.", file=sys.stderr); sys.exit(1)
    if "sslmode=" not in db_url:
        # hosted Postgres usually needs SSL; add if missing
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required.", file=sys.stderr); sys.exit(1)
    return db_url, api_key

def connect_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)

def fetch_total_counts(engine: Engine, company_id: int | None) -> tuple[int, int]:
    with engine.begin() as conn:
        if company_id is None:
            sql = text("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE embedding IS NULL) AS missing
                FROM public.knowledge
            """)
            row = conn.execute(sql).mappings().one()
        else:
            sql = text("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE embedding IS NULL) AS missing
                FROM public.knowledge
                WHERE company_id = :cid
            """)
            row = conn.execute(sql, {"cid": company_id}).mappings().one()
        return int(row["total"]), int(row["missing"])

def fetch_batch(engine: Engine, company_id: int | None, limit: int) -> list[dict]:
    with engine.begin() as conn:
        if company_id is None:
            sql = text("""
                SELECT id, content, title
                FROM public.knowledge
                WHERE embedding IS NULL
                  AND content IS NOT NULL
                ORDER BY id
                LIMIT :lim
            """)
            rows = conn.execute(sql, {"lim": limit}).mappings().all()
        else:
            sql = text("""
                SELECT id, content, title
                FROM public.knowledge
                WHERE embedding IS NULL
                  AND content IS NOT NULL
                  AND company_id = :cid
                ORDER BY id
                LIMIT :lim
            """)
            rows = conn.execute(sql, {"cid": company_id, "lim": limit}).mappings().all()
        return [dict(r) for r in rows]

def update_tsv_for_ids(engine: Engine, ids: Sequence[int]) -> None:
    if not ids: return
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE public.knowledge
                SET tsv = to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
                WHERE id = ANY(:ids)
            """),
            {"ids": list(ids)}
        )

def update_embeddings_rowwise(engine: Engine, ids: Sequence[int], vectors: Sequence[List[float]]) -> None:
    """Row-by-row executemany so the pgvector adapter binds lists → vector."""
    if not ids: return
    payload = [{"id": i, "vec": v} for i, v in zip(ids, vectors)]
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE public.knowledge SET embedding = :vec WHERE id = :id"),
            payload
        )

def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    retries, delay = 3, 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt == retries: raise
            print(f"[warn] embeddings error ({e}); retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay); delay *= 1.7
    raise RuntimeError("unreachable")

def main():
    args = parse_args()
    db_url, api_key = get_env(args)
    engine = connect_engine(db_url)
    client = OpenAI(api_key=api_key)

    total, missing = fetch_total_counts(engine, args.company_id)
    print(f"knowledge rows total={total}, missing_embeddings={missing}, model={args.model}")

    if missing == 0:
        print("Nothing to backfill. Done."); return

    processed = 0
    while True:
        rows = fetch_batch(engine, args.company_id, args.batch)
        if not rows: break

        ids = [int(r["id"]) for r in rows]
        texts = [((r.get("content") or "")[:MAX_INPUT_LEN]) for r in rows]
        nonempty = [(i, t) for i, t in zip(ids, texts) if t.strip()]

        if not nonempty:
            if not args.dry_run:
                update_tsv_for_ids(engine, ids)
            processed += len(ids)
            print(f"Updated TSV for {len(ids)} empty-content rows. Progress≈{processed}/{missing}")
            continue

        masked_ids = [i for i, _ in nonempty]
        masked_texts = [t for _, t in nonempty]

        vectors = embed_batch(client, args.model, masked_texts)

        if args.dry_run:
            print(f"[dry-run] would embed {len(masked_ids)} rows; example ids: {masked_ids[:5]}")
        else:
            update_embeddings_rowwise(engine, masked_ids, vectors)
            update_tsv_for_ids(engine, masked_ids)
            print(f"Embedded+indexed {len(masked_ids)} rows. Last id={masked_ids[-1]}  Progress≈{processed+len(ids)}/{missing}")

        processed += len(ids)

    total2, missing2 = fetch_total_counts(engine, args.company_id)
    print(f"Done. total={total2}, missing_embeddings={missing2}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr); sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

"""
Async LISTEN/NOTIFY consumer for grade-scale conversion audit events.

The ``auto_convert_disease_grade`` trigger (schema/triggers/auto_convert_grading.sql)
emits a ``grade_conversion`` NOTIFY whenever it populates ``scaled_grade``. This module
runs a long-lived listener that consumes those events and records the corresponding
audit row in ``transformation_operations`` + ``provenance_transformations`` using the
existing deterministic, idempotent helpers (``log_and_link_transformation``).

Why a listener instead of logging inside the trigger:
    The old trigger inserted the audit row directly with ``gen_random_uuid()``. Because
    ``upsert_disease_grading`` uses ``ON CONFLICT DO UPDATE``, the BEFORE INSERT/UPDATE
    trigger refired on every idempotent re-ingest and produced a brand-new duplicate row
    each time. Routing the audit write through Python lets us reuse the deterministic
    ``generate_transformation_uuid`` scheme so re-delivered / duplicate events converge.

Durability:
    NOTIFY events are dropped if no listener is connected. This listener is the
    real-time path; ``reconcile_grade_conversions`` (internal.ingest.framework.provenance)
    is the completeness backstop. In production the listener is kept alive by pm2
    (see setup.sh).

Usage:
    # Standalone (this is what pm2 runs):
    #   uv run python -m internal.ingest.framework.provenance_listener

    # Programmatically, wrapping a block of writes:
    async with start_grade_conversion_listener():
        await ingest_something()
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import psycopg

from chaksudb.config.config import db_config
from chaksudb.ingest.framework.transformations import log_and_link_transformation

logger = logging.getLogger(__name__)

CHANNEL = "grade_conversion"
TRANSFORMATION_TYPE = "grade_scale_conversion"


async def record_grade_conversion(event: dict, *, operator: str = "provenance_listener") -> bool:
    """Record one grade-conversion audit row from a parsed event dict.

    Shared by the live listener and the reconciliation sweep so both paths produce
    byte-identical rows. Idempotent: ``log_and_link_transformation`` derives a
    deterministic ``transformation_id`` from the operation type + hashed input/params and
    upserts with ``ON CONFLICT DO NOTHING`` on both the operation and the chain link, so
    duplicate / re-delivered events converge to the same rows.

    Returns True if a row was (idempotently) recorded, False if the event was skipped.
    """
    chain_raw = event.get("provenance_chain_id")
    if not chain_raw:
        # Trigger only emits when provenance_chain_id is present, but guard anyway.
        logger.debug("grade_conversion event without provenance_chain_id; skipping: %r", event)
        return False
    try:
        chain_id = uuid.UUID(str(chain_raw))
    except (ValueError, TypeError):
        logger.warning("grade_conversion event with invalid chain id %r; skipping", chain_raw)
        return False

    # input_data / parameters feed the deterministic UUID, so keep them stable per
    # logical conversion (do NOT include the chain_id here — the link carries that).
    input_data = {
        "grading_id": event.get("grading_id"),
        "image_id": event.get("image_id"),
        "scale_id": event.get("scale_id"),
        "original_grade": event.get("original_grade"),
        "disease_type": event.get("disease_type"),
    }
    output_data = {
        "scaled_grade": event.get("scaled_grade"),
        "target_scale_id": event.get("target_scale_id"),
    }
    parameters = {
        "mode": event.get("mode"),
        "target_scale_name": event.get("target_scale_name"),
    }

    await log_and_link_transformation(
        chain_id=chain_id,
        transformation_type=TRANSFORMATION_TYPE,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters,
        operator=operator,
        notes="Auto grade-scale conversion to ICDR_0_4 (via auto_convert_disease_grade).",
    )
    logger.debug(
        "Recorded grade_scale_conversion for grading_id=%s chain=%s",
        event.get("grading_id"), chain_id,
    )
    return True


async def _handle_event(payload: str) -> None:
    """Parse a NOTIFY payload and record the audit row."""
    try:
        event = json.loads(payload)
    except (ValueError, TypeError) as exc:
        logger.warning("Ignoring malformed grade_conversion payload (%s): %r", exc, payload)
        return
    await record_grade_conversion(event)


async def _open_listen_connection() -> psycopg.AsyncConnection:
    """Open a dedicated autocommit connection and LISTEN on the channel.

    A dedicated (non-pooled) connection is required: pooled connections get recycled,
    which would silently drop the LISTEN registration. Autocommit is required for
    LISTEN to receive notifications promptly.
    """
    conn = await psycopg.AsyncConnection.connect(
        db_config.async_connection_string,
        autocommit=True,
    )
    await conn.execute(f"LISTEN {CHANNEL}")
    logger.info("Listening on '%s' channel for grade-conversion audit events", CHANNEL)
    return conn


async def run_listener(stop_event: Optional[asyncio.Event] = None) -> None:
    """Run the listener loop until ``stop_event`` is set (or forever).

    Reconnects with backoff if the connection drops, so it survives DB blips. Each
    handled event is idempotent, so reprocessing after a reconnect is harmless.
    """
    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    backoff = 1.0
    while not _stopped():
        conn: Optional[psycopg.AsyncConnection] = None
        try:
            conn = await _open_listen_connection()
            backoff = 1.0  # reset after a successful connect
            # Poll in short windows so stop_event is observed promptly without
            # tearing down / re-LISTENing the connection on every idle period.
            while not _stopped():
                async for notify in conn.notifies(timeout=5.0):
                    if notify.channel != CHANNEL:
                        continue
                    try:
                        await _handle_event(notify.payload)
                    except Exception:  # never let one bad event kill the loop
                        logger.exception(
                            "Failed to record grade_conversion event: %r", notify.payload
                        )
                    if _stopped():
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("grade_conversion listener connection error; reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        finally:
            if conn is not None:
                await conn.close()


@asynccontextmanager
async def start_grade_conversion_listener() -> AsyncGenerator[asyncio.Task, None]:
    """Run the listener as a background task for the duration of the context."""
    stop_event = asyncio.Event()
    task = asyncio.create_task(run_listener(stop_event))
    try:
        yield task
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting grade-conversion provenance listener (pid via pm2)")
    try:
        asyncio.run(run_listener())
    except KeyboardInterrupt:
        logger.info("Listener stopped by KeyboardInterrupt")


if __name__ == "__main__":
    main()

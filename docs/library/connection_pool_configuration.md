# Database Connection Pool Configuration Guide

## Overview

The connection pool configuration (`min_connections` and `max_connections`) is critical for database performance and avoiding connection exhaustion. This guide explains how to set these values appropriately.

## Configuration

Connection pool settings are configured via environment variables (with `DB_` prefix) or defaults in `chaksudb/config/config.py`:

```bash
# In .env file
DB_MIN_CONNECTIONS=2
DB_MAX_CONNECTIONS=10
```

## Key Factors to Consider

### 1. **PostgreSQL Server Limits**

**First, check your PostgreSQL `max_connections` setting:**

```sql
SHOW max_connections;
```

**Rule of thumb:** Your application's `max_connections` should be **≤ 80% of PostgreSQL's `max_connections`** to leave room for:
- Admin connections
- Other applications
- Maintenance operations
- Connection overhead

**Example:**
- PostgreSQL `max_connections` = 100
- Recommended app `max_connections` = 80 (or less)

### 2. **Concurrent Operations**

**For ingestion workloads (your use case):**

```
max_connections = (number of concurrent ingestions) + (headroom for other operations)
```

**Current setup:**
- You have 50+ ingest scripts
- With semaphore limiting to `max_connections - 2`
- If `max_connections = 10`, then 8 concurrent ingestions

**Recommendation for full ingestion:**
- If running all 36 scripts: `max_connections = 20-30` (allows 18-28 concurrent)
- If running smaller batches: `max_connections = 10-15` is fine

### 3. **Memory Usage**

Each PostgreSQL connection uses memory:
- **Per connection:** ~2-10 MB (depends on `work_mem` and query complexity)
- **Total memory:** `max_connections × per_connection_memory`

**Example:**
- `max_connections = 20`
- Per connection: ~5 MB
- Total: ~100 MB (acceptable for most systems)

### 4. **Workload Type**

**I/O-bound workloads (your ingestion):**
- Many concurrent connections are beneficial
- Connections spend time waiting for disk I/O
- **Higher `max_connections` is better** (20-50)

**CPU-bound workloads:**
- Fewer connections needed
- Connections compete for CPU
- **Lower `max_connections` is better** (5-10)

### 5. **min_connections vs max_connections**

**`min_connections`:**
- Connections kept alive in the pool
- Reduces connection establishment overhead
- **Recommended:** 2-5 (keeps pool "warm")

**`max_connections`:**
- Maximum connections the pool can create
- Should be based on concurrent operations
- **Recommended:** See formulas below

## Recommended Formulas

### For Ingestion Workloads

```python
# Formula 1: Based on concurrent ingestions
max_connections = (number_of_concurrent_ingestions) + 2-5

# Formula 2: Based on PostgreSQL limits
max_connections = min(
    (postgresql_max_connections * 0.8),  # 80% of PostgreSQL limit
    (number_of_concurrent_ingestions + 5)  # Your needs + headroom
)

# Formula 3: Based on system resources
max_connections = min(
    (available_memory_mb / 5),  # Assuming 5 MB per connection
    (postgresql_max_connections * 0.8)
)
```

### For Production

```python
# For batch processing
max_connections = (number_of_worker_threads) + 2
```

## Configuration Examples

### Example 1: Full Database Setup (36 concurrent ingestions)

```bash
# .env file
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=30
```

**Reasoning:**
- 30 max allows ~28 concurrent ingestions (with semaphore limit)
- 5 min keeps pool warm
- Leaves room for other operations

### Example 2: Single Dataset Ingestion

```bash
DB_MIN_CONNECTIONS=2
DB_MAX_CONNECTIONS=10
```

**Reasoning:**
- 10 max is sufficient for single ingestion
- 2 min is minimal overhead

### Example 3: Production API Server

```bash
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=20
```

**Reasoning:**
- Handles multiple concurrent API requests
- Keeps pool warm with 5 min connections

### Example 4: Resource-Constrained System

```bash
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=5
```

**Reasoning:**
- Limited memory/system resources
- Minimal connection overhead

## Monitoring and Tuning

### Check Current Usage

```sql
-- See active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- See all connections
SELECT count(*) FROM pg_stat_activity;

-- See connections by application
SELECT application_name, count(*) 
FROM pg_stat_activity 
GROUP BY application_name;
```

### Signs You Need to Adjust

**Increase `max_connections` if:**
- ❌ Seeing "connection pool exhausted" errors
- ❌ High wait times for connections
- ❌ Many tasks waiting for database connections

**Decrease `max_connections` if:**
- ❌ PostgreSQL hitting `max_connections` limit
- ❌ High memory usage
- ❌ Connections idle most of the time

**Adjust `min_connections` if:**
- ❌ High connection establishment overhead
- ❌ Cold start performance issues (increase)
- ❌ Unnecessary memory usage (decrease)

## PostgreSQL Server Configuration

Also check your PostgreSQL server settings:

```sql
-- Check current settings
SHOW max_connections;
SHOW shared_buffers;
SHOW work_mem;

-- Recommended for ingestion workloads
-- max_connections = 100-200 (depending on system)
-- shared_buffers = 25% of RAM
-- work_mem = (RAM - shared_buffers) / (max_connections * 2)
```

## Best Practices

1. **Start conservative:** Begin with lower values and increase as needed
2. **Monitor first:** Check connection usage before optimizing
3. **Leave headroom:** Never use 100% of PostgreSQL's `max_connections`
4. **Match workload:** Adjust based on your actual concurrent operations
5. **Test changes:** Monitor performance after changing pool size

## Quick Reference Table

| Use Case | min_connections | max_connections | Notes |
|----------|----------------|------------------|-------|
| Full ingestion (50+ datasets) | 5 | 30 | High concurrency needed |
| Single dataset ingestion | 2 | 10 | Standard setup |
| Production API | 5 | 20 | Multiple concurrent requests |
| Development/Testing | 1 | 5 | Minimal resources |
| Resource-constrained | 1 | 5 | Limited system resources |

## Current Setup Analysis

Based on your current code:

```python
# In setup_full_database.py
max_concurrent = max(1, db_config.max_connections - 2)
```

**If `max_connections = 10`:**
- Semaphore allows 8 concurrent ingestions
- 2 connections reserved for other operations
- **Recommendation:** Increase to 20-30 for full ingestion

**If `max_connections = 30`:**
- Semaphore allows 28 concurrent ingestions
- Good for running all 50+ datasets (some will queue)
- **This is a good configuration for full ingestion**

## Summary

**For your ingestion workload:**
1. Set `DB_MAX_CONNECTIONS=30` in `.env` (or higher if PostgreSQL allows)
2. Set `DB_MIN_CONNECTIONS=5` to keep pool warm
3. Monitor connection usage during ingestion
4. Adjust based on actual performance

**Quick setup:**
```bash
# Add to .env file
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=30
```

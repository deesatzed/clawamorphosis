-- CLAW — Database Schema
-- SQLite with WAL mode, FTS5, and sqlite-vec

-- 1. PROJECTS
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    tech_stack TEXT NOT NULL DEFAULT '{}',       -- JSON string
    project_rules TEXT,
    banned_dependencies TEXT NOT NULL DEFAULT '[]', -- JSON array string
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- 2. TASKS (Work Queue)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING','EVALUATING','PLANNING','DISPATCHED','CODING','REVIEWING','STUCK','DONE')),
    priority INTEGER NOT NULL DEFAULT 0,
    task_type TEXT,
    recommended_agent TEXT,
    assigned_agent TEXT,
    context_snapshot_id TEXT,
    attempt_count INTEGER DEFAULT 0,
    escalation_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(project_id, priority DESC);

-- 3. HYPOTHESIS_LOG (Trial & Error Memory)
CREATE TABLE IF NOT EXISTS hypothesis_log (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    approach_summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'FAILURE'
        CHECK (outcome IN ('SUCCESS','FAILURE')),
    error_signature TEXT,
    error_full TEXT,
    files_changed TEXT NOT NULL DEFAULT '[]',     -- JSON array string
    duration_seconds REAL,
    model_used TEXT,
    agent_id TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(task_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS idx_hyp_task ON hypothesis_log(task_id);
CREATE INDEX IF NOT EXISTS idx_hyp_error_sig ON hypothesis_log(error_signature);

-- 4. METHODOLOGIES (Long-Term Memory / RAG)
CREATE TABLE IF NOT EXISTS methodologies (
    id TEXT PRIMARY KEY,
    problem_description TEXT NOT NULL,
    solution_code TEXT NOT NULL,
    methodology_notes TEXT,
    source_task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    tags TEXT NOT NULL DEFAULT '[]',               -- JSON array string
    language TEXT,
    scope TEXT NOT NULL DEFAULT 'project',
    methodology_type TEXT,
    files_affected TEXT NOT NULL DEFAULT '[]',      -- JSON array string
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- MEE lifecycle fields
    lifecycle_state TEXT NOT NULL DEFAULT 'viable'
        CHECK (lifecycle_state IN ('embryonic','viable','thriving','declining','dormant','dead')),
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved_at TEXT,
    generation INTEGER NOT NULL DEFAULT 0,
    fitness_vector TEXT NOT NULL DEFAULT '{}',      -- JSON string
    parent_ids TEXT NOT NULL DEFAULT '[]',          -- JSON array string
    superseded_by TEXT,
    prism_data TEXT,                                  -- JSON: PrismEmbedding (nullable)
    capability_data TEXT,                              -- JSON: CapabilityData (nullable)
    novelty_score REAL,                                -- 0.0-1.0: how different from existing KB
    potential_score REAL                                -- 0.0-1.0: future composability/value
);
CREATE INDEX IF NOT EXISTS idx_meth_scope ON methodologies(scope);
CREATE INDEX IF NOT EXISTS idx_meth_lifecycle ON methodologies(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_meth_novelty ON methodologies(novelty_score DESC);

-- Methodology embeddings (sqlite-vec virtual table)
-- Stores 384-dimensional float32 vectors for semantic search
-- Queried as: SELECT rowid, distance FROM methodology_embeddings WHERE embedding MATCH ?
CREATE VIRTUAL TABLE IF NOT EXISTS methodology_embeddings USING vec0(
    methodology_id TEXT PRIMARY KEY,
    embedding float[384]
);

-- Methodology full-text search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS methodology_fts USING fts5(
    methodology_id UNINDEXED,
    problem_description,
    methodology_notes,
    tags
);

-- 5. PEER_REVIEWS (Escalation Diagnoses)
CREATE TABLE IF NOT EXISTS peer_reviews (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    model_used TEXT NOT NULL,
    diagnosis TEXT NOT NULL,
    recommended_approach TEXT,
    reasoning TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_peer_task ON peer_reviews(task_id);

-- 6. CONTEXT_SNAPSHOTS (Checkpoint/Rewind State)
CREATE TABLE IF NOT EXISTS context_snapshots (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    git_ref TEXT NOT NULL,
    file_manifest TEXT,                            -- JSON string
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_snap_task ON context_snapshots(task_id);

-- 7. METHODOLOGY_LINKS (Stigmergic co-retrieval)
CREATE TABLE IF NOT EXISTS methodology_links (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES methodologies(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES methodologies(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL DEFAULT 'co_retrieval',
    strength REAL NOT NULL DEFAULT 1.0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source_id, target_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_meth_links_source ON methodology_links(source_id);
CREATE INDEX IF NOT EXISTS idx_meth_links_target ON methodology_links(target_id);

-- 8. TOKEN_COSTS (Per-call LLM cost tracking)
CREATE TABLE IF NOT EXISTS token_costs (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    run_id TEXT,
    agent_role TEXT NOT NULL DEFAULT '',
    agent_id TEXT,
    model_used TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_token_costs_task ON token_costs(task_id);
CREATE INDEX IF NOT EXISTS idx_token_costs_agent ON token_costs(agent_id);
CREATE INDEX IF NOT EXISTS idx_token_costs_created ON token_costs(created_at DESC);

-- =========================================================================
-- CLAW-specific tables (not in ralfed)
-- =========================================================================

-- 9. AGENT_SCORES (Bayesian routing scores per task_type + agent)
CREATE TABLE IF NOT EXISTS agent_scores (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    total_attempts INTEGER NOT NULL DEFAULT 0,
    avg_duration_seconds REAL NOT NULL DEFAULT 0.0,
    avg_quality_score REAL NOT NULL DEFAULT 0.0,
    avg_cost_usd REAL NOT NULL DEFAULT 0.0,
    last_used_at TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(agent_id, task_type)
);
CREATE INDEX IF NOT EXISTS idx_agent_scores_agent ON agent_scores(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_scores_type ON agent_scores(task_type);

-- 10. PROMPT_VARIANTS (A/B testing for prompt evolution)
CREATE TABLE IF NOT EXISTS prompt_variants (
    id TEXT PRIMARY KEY,
    prompt_name TEXT NOT NULL,
    variant_label TEXT NOT NULL DEFAULT 'control',
    content TEXT NOT NULL,
    agent_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    avg_quality_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(prompt_name, variant_label, agent_id)
);

-- 11. CAPABILITY_BOUNDARIES (Tasks that all agents fail)
CREATE TABLE IF NOT EXISTS capability_boundaries (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    task_description TEXT NOT NULL,
    agents_attempted TEXT NOT NULL DEFAULT '[]',    -- JSON array string
    failure_signatures TEXT NOT NULL DEFAULT '[]',  -- JSON array string
    discovered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_retested_at TEXT,
    retest_result TEXT,
    escalated_to_human INTEGER NOT NULL DEFAULT 0,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cap_bounds_type ON capability_boundaries(task_type);

-- 12. FLEET_REPOS (Fleet repo tracking)
CREATE TABLE IF NOT EXISTS fleet_repos (
    id TEXT PRIMARY KEY,
    repo_path TEXT NOT NULL UNIQUE,
    repo_name TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','evaluating','enhancing','completed','failed','skipped')),
    enhancement_branch TEXT,
    last_evaluated_at TEXT,
    evaluation_score REAL,
    budget_allocated_usd REAL NOT NULL DEFAULT 0.0,
    budget_used_usd REAL NOT NULL DEFAULT 0.0,
    tasks_created INTEGER NOT NULL DEFAULT 0,
    tasks_completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_fleet_repos_status ON fleet_repos(status);
CREATE INDEX IF NOT EXISTS idx_fleet_repos_priority ON fleet_repos(priority DESC);

-- 13. EPISODES (Episodic memory — session event log)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL DEFAULT '{}',          -- JSON string
    agent_id TEXT,
    task_id TEXT,
    cycle_level TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(event_type);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);

-- 14. SYNERGY_EXPLORATION_LOG (Tracks explored capability pairs — SMART dedup)
CREATE TABLE IF NOT EXISTS synergy_exploration_log (
    id TEXT PRIMARY KEY,
    cap_a_id TEXT NOT NULL,
    cap_b_id TEXT NOT NULL,
    explored_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    result TEXT NOT NULL DEFAULT 'pending'
        CHECK (result IN ('pending','synergy','no_match','error','stale')),
    synergy_score REAL,
    synergy_type TEXT,
    edge_id TEXT,
    exploration_method TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    UNIQUE(cap_a_id, cap_b_id)
);
CREATE INDEX IF NOT EXISTS idx_synergy_log_cap_a ON synergy_exploration_log(cap_a_id);
CREATE INDEX IF NOT EXISTS idx_synergy_log_cap_b ON synergy_exploration_log(cap_b_id);
CREATE INDEX IF NOT EXISTS idx_synergy_log_result ON synergy_exploration_log(result);

-- 15. GOVERNANCE_LOG (Audit trail for governance actions)
CREATE TABLE IF NOT EXISTS governance_log (
    id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    methodology_id TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_governance_log_action ON governance_log(action_type);
CREATE INDEX IF NOT EXISTS idx_governance_log_created ON governance_log(created_at DESC);

#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Demo 2: "The Heist" — Cross-Repo Knowledge Transfer
# =============================================================================
#
# CLAW mines the ralfed reference implementation for patterns, stores them
# as 384-dimensional vector embeddings in semantic memory, then generates
# enhancement tasks for CLAW itself.
#
# Knowledge stolen from one codebase becomes actionable improvements
# in another. No other open-source tool does this.
#
# Prerequisites: OPENROUTER_API_KEY environment variable set
# Runtime: ~1-3 minutes (1 LLM call + embedding generation)
# =============================================================================

BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
DIM="\033[2m"
RESET="\033[0m"

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Demo 2: THE HEIST — Cross-Repo Knowledge Transfer${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  CLAW will mine the ${CYAN}ralfed${RESET} reference implementation to extract"
echo -e "  transferable patterns, features, and architectural ideas."
echo ""
echo -e "  The pipeline:"
echo -e "    1. Serialize ralfed's source code (~47 Python files)"
echo -e "    2. Send to LLM with a mining-specific prompt"
echo -e "    3. Parse structured JSON findings (title, category, relevance)"
echo -e "    4. Store each finding as a 384-dim vector embedding in semantic memory"
echo -e "    5. Generate enhancement tasks for CLAW from high-relevance findings"
echo ""
echo -e "  ${DIM}A human doing this cross-pollination manually needs days.${RESET}"
echo -e "  ${DIM}CLAW does it in under 2 minutes.${RESET}"
echo ""

# --- Preflight checks ---

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY is not set.${RESET}"
    echo ""
    echo "  Set it with:"
    echo "    export OPENROUTER_API_KEY=sk-or-v1-your-key-here"
    echo ""
    echo "  Get a key at: https://openrouter.ai/keys"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${CYAN}Preflight checks...${RESET}"

if ! command -v claw &>/dev/null; then
    echo -e "  ${YELLOW}Installing CLAW...${RESET}"
    pip install -e . -q 2>/dev/null
fi
echo -e "  ${GREEN}claw CLI: installed${RESET}"
echo -e "  ${GREEN}OPENROUTER_API_KEY: set${RESET}"

# Check ralfed exists
if [ ! -d "ralfed" ]; then
    echo -e "  ${RED}ralfed/ directory not found — cannot mine${RESET}"
    echo -e "  ${DIM}This demo requires the ralfed reference implementation in the repo root.${RESET}"
    exit 1
fi
echo -e "  ${GREEN}ralfed/: found (reference implementation)${RESET}"

# Fresh database (remove WAL/SHM journals too)
rm -f data/claw.db data/claw.db-wal data/claw.db-shm
mkdir -p data
echo -e "  ${GREEN}Database: reset (fresh start)${RESET}"

echo ""
echo -e "${BOLD}Phase 1: Mining ralfed for patterns...${RESET}"
echo -e "${DIM}(Serializing source files, calling LLM, parsing findings)${RESET}"
echo ""

# --- Mine ralfed ---

claw mine ./ralfed --target . --max-repos 1 --min-relevance 0.5

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 2: Verifying semantic memory storage...${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

# Show what's in the database
FINDING_COUNT=$(sqlite3 data/claw.db "SELECT COUNT(*) FROM methodologies WHERE tags LIKE '%mined%'" 2>/dev/null || echo "0")
TASK_COUNT=$(sqlite3 data/claw.db "SELECT COUNT(*) FROM tasks WHERE title LIKE '%Mined%'" 2>/dev/null || echo "0")

echo -e "  Findings stored in semantic memory: ${GREEN}${FINDING_COUNT}${RESET}"
echo -e "  Enhancement tasks generated:        ${GREEN}${TASK_COUNT}${RESET}"
echo ""

if [ "$FINDING_COUNT" -gt 0 ]; then
    echo -e "${CYAN}Stored methodologies (with 384-dim embeddings):${RESET}"
    sqlite3 -header -column data/claw.db \
        "SELECT substr(problem_description, 1, 70) as finding, scope, methodology_type as type, lifecycle_state as lifecycle FROM methodologies WHERE tags LIKE '%mined%' ORDER BY created_at" 2>/dev/null || true
    echo ""
fi

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 3: Viewing generated tasks...${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

claw results

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  What just happened${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  1. CLAW serialized ralfed's source code (skipping .git, __pycache__, etc.)"
echo -e "  2. Sent the serialized repo to an LLM with a mining-specific prompt"
echo -e "  3. The LLM returned structured JSON findings with:"
echo -e "     - Title, description, category, source files"
echo -e "     - Implementation sketch (how to adapt for CLAW)"
echo -e "     - Relevance score (0.4 - 1.0)"
echo -e "  4. Each finding was stored in semantic memory as a Methodology:"
echo -e "     - scope=global, type=PATTERN, lifecycle=embryonic"
echo -e "     - 384-dimensional vector embedding via all-MiniLM-L6-v2"
echo -e "     - Tags: [mined, source:ralfed, category:...]"
echo -e "  5. Findings with relevance >= 0.5 became enhancement tasks"
echo -e "     - Each task has a recommended agent (claude/codex/gemini/grok)"
echo -e "     - Priority mapped from relevance score"
echo ""
echo -e "  ${GREEN}The knowledge transfer is REAL.${RESET}"
echo -e "  ${GREEN}Future tasks can query semantic memory and retrieve these patterns.${RESET}"
echo -e "  ${GREEN}Run 'claw enhance .' to have agents work on the generated tasks.${RESET}"
echo ""
echo -e "  ${DIM}Database: data/claw.db${RESET}"
echo -e "  ${DIM}Next: try Demo 3 (The Gauntlet) for the full autonomous cycle${RESET}"
echo ""

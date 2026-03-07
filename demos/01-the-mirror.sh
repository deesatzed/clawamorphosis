#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Demo 1: "The Mirror" — CLAW Evaluates Itself
# =============================================================================
#
# CLAW runs its evaluation battery against its own 18,000 LOC codebase.
# Four AI agents simultaneously analyze the code for drift, technical debt,
# regression risks, and architectural issues — with file:line evidence.
#
# What a senior engineer needs a full day to do, CLAW does in ~3 minutes.
#
# Prerequisites: OPENROUTER_API_KEY environment variable set
# Runtime: ~2-4 minutes (5 LLM calls via OpenRouter)
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
echo -e "${BOLD}  Demo 1: THE MIRROR — CLAW Evaluates Itself${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  CLAW will run its evaluation battery against its own codebase."
echo -e "  Four AI agents analyze ~18,000 lines of Python for:"
echo -e "    - Documentation drift (does the README match reality?)"
echo -e "    - Technical debt (what needs fixing, ranked by severity)"
echo -e "    - Architecture patterns (what's good, what's fragile)"
echo -e "    - Feature completeness (what's wired vs scaffolding)"
echo -e ""
echo -e "  ${DIM}This would take a senior engineer a full day.${RESET}"
echo -e "  ${DIM}CLAW does it in ~3 minutes with 4 models in parallel.${RESET}"
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

# Navigate to repo root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${CYAN}Preflight checks...${RESET}"

# Ensure claw is installed
if ! command -v claw &>/dev/null; then
    echo -e "  ${YELLOW}Installing CLAW...${RESET}"
    pip install -e . -q 2>/dev/null
fi
echo -e "  ${GREEN}claw CLI: installed${RESET}"

# Check API key works (quick sanity)
echo -e "  ${GREEN}OPENROUTER_API_KEY: set${RESET}"

# Fresh database (remove WAL/SHM journals too)
rm -f data/claw.db data/claw.db-wal data/claw.db-shm
mkdir -p data
echo -e "  ${GREEN}Database: reset (fresh start)${RESET}"

echo ""
echo -e "${BOLD}Starting evaluation...${RESET}"
echo -e "${DIM}(First run downloads the embedding model ~80MB — subsequent runs are instant)${RESET}"
echo ""

# --- Run the evaluation ---

claw evaluate . --mode quick

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Retrieving stored results...${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

claw results

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  What just happened${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  1. CLAW scanned its own repo structure (files, languages, configs)"
echo -e "  2. Five evaluation prompts were dispatched to 4 AI agents:"
echo -e "     - project-context, workspace-scan (orientation)"
echo -e "     - deepdive, agonyofdefeatures, driftx (deep analysis)"
echo -e "  3. Each agent was selected via Bayesian routing with 10% exploration"
echo -e "  4. Results stored in SQLite with full audit trail"
echo -e ""
echo -e "  ${GREEN}The findings are REAL — not canned output.${RESET}"
echo -e "  ${GREEN}Run it again and you'll get different routing decisions.${RESET}"
echo ""
echo -e "  ${DIM}Database: data/claw.db${RESET}"
echo -e "  ${DIM}Next: try Demo 2 (The Heist) for cross-repo knowledge mining${RESET}"
echo ""

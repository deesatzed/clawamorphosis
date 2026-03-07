#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Demo 3: "The Gauntlet" — Full Claw Cycle with Verification Gate
# =============================================================================
#
# CLAW runs the complete autonomous pipeline on a real task:
#   grab -> evaluate -> decide -> act -> verify -> learn
#
# A custom goal is injected, then CLAW:
#   1. Routes it to the best-fit agent via Bayesian scoring
#   2. The agent produces a solution via OpenRouter
#   3. The 7-check verification gate audits the output:
#      - Dependency jail, style match, chaos check, placeholder scan,
#        drift alignment (384-dim cosine similarity), claim validation
#   4. If approved: pattern saved to semantic memory, agent score updated
#      If rejected: failure logged to error KB — BOTH outcomes prove it works
#
# Prerequisites: OPENROUTER_API_KEY environment variable set
# Runtime: ~2-4 minutes (2-3 LLM calls + verification)
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
echo -e "${BOLD}  Demo 3: THE GAUNTLET — Full Claw Cycle${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  CLAW will run the complete autonomous pipeline:"
echo ""
echo -e "    ${CYAN}grab${RESET}     Pull the highest-priority task from the queue"
echo -e "    ${CYAN}evaluate${RESET}  Query error KB + semantic memory for context"
echo -e "    ${CYAN}decide${RESET}    Bayesian routing selects the best agent"
echo -e "    ${CYAN}act${RESET}       The chosen agent produces a solution"
echo -e "    ${CYAN}verify${RESET}    7-check verification gate audits the output"
echo -e "    ${CYAN}learn${RESET}     Update agent scores, save patterns to memory"
echo ""
echo -e "  The verification gate runs 7 independent checks:"
echo -e "    1. Dependency jail    — no banned imports or destructive code"
echo -e "    2. Style match        — consistent formatting, no wildcard imports"
echo -e "    3. Chaos check        — no bare except, no eval(), no hardcoded creds"
echo -e "    4. Placeholder scan   — no TODO, FIXME, NotImplementedError left behind"
echo -e "    5. Drift alignment    — semantic similarity via 384-dim embeddings"
echo -e "    6. Claim validation   — assertions must have evidence"
echo -e "    7. LLM deep review    — optional second-opinion from another model"
echo ""
echo -e "  ${DIM}If the gate approves: success pattern stored, agent score updated.${RESET}"
echo -e "  ${DIM}If the gate rejects: failure logged — BOTH outcomes prove CLAW works.${RESET}"
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

# Fresh database (remove WAL/SHM journals too)
rm -f data/claw.db data/claw.db-wal data/claw.db-shm
mkdir -p data
echo -e "  ${GREEN}Database: reset (fresh start)${RESET}"

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 1: Injecting a goal${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

# Inject a real, useful task
claw add-goal . \
    --title "Analyze error handling patterns and suggest improvements" \
    --description "Review the CLAW codebase error handling strategy across all modules. Identify: (1) bare except clauses that swallow errors silently, (2) inconsistent error logging patterns, (3) missing error context in exception chains, (4) opportunities to use custom exception types from core/exceptions.py instead of generic Exception. Provide specific file:line references and concrete improvement suggestions." \
    --priority high \
    --type analysis

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 2: Running the full Claw cycle${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "${DIM}Watch the cycle steps: grab -> evaluate -> decide -> act -> verify -> learn${RESET}"
echo ""

# Run the full pipeline (1 task, autonomous mode)
claw enhance . --max-tasks 1 --mode autonomous

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 3: Inspecting results${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

claw results

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  Phase 4: System state after learning${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

claw status

echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  What just happened${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  1. ${CYAN}Goal injected${RESET}: 'Analyze error handling patterns'"
echo -e "     Priority: high (8), Type: analysis, recommended agent: auto-routed"
echo ""
echo -e "  2. ${CYAN}Claw cycle executed${RESET}:"
echo -e "     - grab:     Pulled the highest-priority task from SQLite queue"
echo -e "     - evaluate:  Queried error KB for forbidden approaches,"
echo -e "                  queried semantic memory for similar past solutions"
echo -e "     - decide:    Dispatcher consulted agent_scores table,"
echo -e "                  applied Bayesian routing with 10% exploration"
echo -e "     - act:       Selected agent called OpenRouter API,"
echo -e "                  received structured analysis response"
echo -e "     - verify:    7-check verification gate ran all checks"
echo -e "     - learn:     Agent score updated, patterns saved to memory"
echo ""
echo -e "  3. ${CYAN}Verification gate${RESET} ran these checks:"
echo -e "     - Dependency jail:  scanned for banned imports"
echo -e "     - Style match:     checked formatting consistency"
echo -e "     - Chaos check:     scanned for dangerous patterns"
echo -e "     - Placeholder scan: searched for TODO/FIXME stubs"
echo -e "     - Drift alignment: cosine similarity of task vs output embeddings"
echo -e "     - Claim validation: checked evidence for assertions"
echo ""
echo -e "  4. ${CYAN}Outcome${RESET}:"
echo -e "     - If approved: pattern stored in semantic memory for future tasks"
echo -e "     - If rejected: failure logged in error KB with root cause"
echo -e "     - Either way: agent score updated for smarter future routing"
echo ""
echo -e "  ${GREEN}This is the complete autonomous pipeline.${RESET}"
echo -e "  ${GREEN}No human in the loop. Real LLM calls. Real verification.${RESET}"
echo -e "  ${GREEN}The system learns from every outcome.${RESET}"
echo ""
echo -e "  ${DIM}Database: data/claw.db${RESET}"
echo -e "  ${DIM}Run again to see different routing (10% exploration randomness)${RESET}"
echo ""

#!/bin/bash
# Double-click launcher: starts Ollama (if needed), starts the Grant Advisor
# server (if needed), opens Firefox to it, then tails the live server logs.

cd "$(dirname "$0")"

LOG_FILE="/tmp/grant_advisor_server.log"
OLLAMA_LOG_FILE="/tmp/grant_advisor_ollama.log"

echo "Grant Advisor launcher"
echo "======================"
echo ""

# 1. Make sure Ollama is running.
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Ollama: already running."
else
  echo "Ollama: starting..."
  ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &
  for i in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama did not come up after 30 seconds -- check $OLLAMA_LOG_FILE"
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
  fi
  echo "Ollama: up."
fi

# 2. Make sure the Grant Advisor server is running.
if curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
  echo "Grant Advisor server: already running."
else
  echo "Grant Advisor server: starting..."
  source .venv/bin/activate
  nohup uvicorn app.server:app --port 8000 > "$LOG_FILE" 2>&1 &
  for i in $(seq 1 30); do
    curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    echo "Server did not come up after 30 seconds -- check $LOG_FILE"
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
  fi
  echo "Grant Advisor server: up."
fi

# 3. Open the app in Firefox.
echo ""
echo "Opening http://127.0.0.1:8000 in Firefox..."
open -a "Firefox" "http://127.0.0.1:8000"

# 4. Stream the live server logs in this window. The server itself keeps
# running in the background even if this window is closed.
echo ""
echo "Live server logs (close this window any time -- the server keeps running):"
echo "----------------------------------------------------------------------"
tail -f "$LOG_FILE"

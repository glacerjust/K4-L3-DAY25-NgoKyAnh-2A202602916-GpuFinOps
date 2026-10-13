import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import CostTracker, BudgetExceeded

t = CostTracker(budgets={"team-chat": 0.05, "team-eval": 100.0})
for i in range(40):
    try:
        t.complete("team-chat", "large", "Summarize this very long document " * 30)
    except BudgetExceeded as e:
        print(f"BLOCKED after {i} chat requests: {e}")
        break
for i in range(5):
    t.complete("team-eval", "small", "classify: positive or negative?", batch=True)

print("\nper-key spend:", t.report())
print("requests logged:", len(t.log))

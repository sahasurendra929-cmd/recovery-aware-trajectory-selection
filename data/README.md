# Data policy and schema

Raw benchmark trajectories are not committed. Download them from their original source and retain the source license/citation.

The pilot expects records with:

```json
{
  "task_id": 42,
  "reward": 1.0,
  "trial": 0,
  "traj": [
    {"role": "assistant", "tool_calls": [{"function": {"name": "tool", "arguments": {}}}]},
    {"role": "tool", "content": "..."}
  ]
}
```

The selection runner saves only trace IDs and summary statistics. Future derived data releases must document consent, source license, redaction policy, and whether records contain simulated or real user information.

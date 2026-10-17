# Security

## Reporting a problem

Report a vulnerability privately through GitHub: the repository's **Security** tab, then **Report a
vulnerability**. Please do not open a public issue for it. Say what you found, how to reproduce it, and
what it exposes.

Problems in scope include a key that could leave the machine it was set on, a task container that can
reach the network or the grader's files, and a way to raise a reward without doing the task (a grader
exploit). For a grader exploit, a broken deliverable that scores higher than it should is the most useful
report: it becomes a new control (docs/tasks.md).

## How keys are handled

- **Where keys live.** Only in the environment, set in the shell or read from `.env` at the repo root
  (`.env.example` lists every variable). Git ignores `.env`, and `uv run evals doctor` fails if it is
  tracked or not ignored, and warns if other users can read it.
- **Never in a config.** A config entry names the variable that holds a key (`key_env`), never the key.
  Loading a file under `configs/` fails when a value looks like a key: a field named like one, a value in
  a provider's key format, or the value of a key variable set in this environment. The error names the
  field, never the value.
- **Never printed or written.** The `evals` command masks the value of every `*_KEY`, `*_TOKEN`,
  `*_SECRET` and `*_PASSWORD` variable in everything it prints, including errors and tracebacks, and in
  every file it writes: `run.json`, `records.jsonl`, each trial's `judge.json`, trajectories and the
  report pages.
- **In a container only when a harness needs it there.** Image models and our own agents run on the host
  and only exchange files with the task's container. A vendor harness that runs inside the container, such
  as Codex, gets its provider's key from Harbor's environment, and the container may then reach only that
  provider's API, and only while the agent runs (`allow_hosts`). A subject's `agent_env`, which Harbor
  saves in its job folder, is refused if a name in it looks like a key.
- **Never on the GPU host.** camera-eval keeps the judges' keys on this machine. The GPU host it rents or
  reaches over SSH gets frames and a token generated at each start, through an SSH tunnel, and no port is
  opened on it.

If a key ever does leak (into a commit, a log or a shared page), revoke it with its provider and issue a
new one; removing it from the file is not enough.

## Isolation of the tasks

Every task gives the agent no network (`network_mode = "no-network"`) and grades in a fresh copy of the
environment that receives only the declared artifacts (`environment_mode = "separate"`). Harbor 0.24
enforces the network rules with nftables, which needs `CONFIG_NFT_FIB_INET` in the Docker engine's kernel;
Docker Desktop on macOS lacks it, and there `evals run` falls back to copies of the tasks with the network
open, saying so in `run.json` and the report. Run on a Linux Docker host, OrbStack or Colima for enforced
isolation. `uv run evals doctor` checks the engine.

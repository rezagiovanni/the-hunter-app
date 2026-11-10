# The Hunter

An Indodax trading bot with a local dashboard. Runs in **paper mode**: it trades
a simulated balance and places no real orders.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/rezagiovanni/the-hunter-app/main/install.sh | sh
```

Then:

```sh
hunter
```

Enter your Indodax API key when prompted. Nothing runs until you tell it to.

## Requirements

- Linux (x86_64 or aarch64)
- Python 3.11 or newer
- `python3-venv`, `curl`, `tar`, and optionally `cron`

The installer checks all of these first and prints the exact command for
anything missing. It never uses `sudo`.

## What the installer does

It installs the code, builds a virtual environment from pinned dependencies,
and registers six scheduled jobs — **all disabled**.
It fetches no market data and places no trades.

## First run

1. `hunter` — enter your API key. It is validated against Indodax and saved to
   `config.yaml` with mode 0600.
2. Open the dashboard at the URL it prints.
3. Go to **Settings** and run the setup actions in order: fetch the pairs, deep
   backfill, fill gaps.
4. Enable the jobs you want: `hunter cron enable market`, then `ohlcv`.
5. When you are ready to trade in paper mode: `hunter start`.

## Commands

| Command | What it does |
|---|---|
| `hunter` | status, or first-time setup if unconfigured |
| `hunter setup` | enter or replace your API key |
| `hunter status` | hunting mode, balance, trades, data, jobs, dashboard |
| `hunter start` / `hunter stop` | start or stop hunting |
| `hunter doctor` | diagnose a broken install and name the fault |
| `hunter cron install` / `enable <job>` / `disable <job>` | manage scheduled jobs |
| `hunter upgrade` | replace the app; data and settings are kept |
| `hunter uninstall` | remove it; trade history is kept unless you say otherwise |

## Where things live

```
~/.local/share/the-hunter/
├── app/          the program
├── data/         state, trade history, market data
├── config.yaml   your settings and credentials
└── logs/         one log per job
```

An upgrade replaces `app/` only. `config.yaml` and `data/` are never touched.

## Configuration

Settings live in `config.yaml`. Every key is optional; deleting one restores its
default. See `config.yaml.example` in the app directory for the full list with
explanations.

## Status

Early. Paper mode only — there is no live-trading executor yet.

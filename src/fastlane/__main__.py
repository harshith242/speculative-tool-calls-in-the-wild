"""FastLane CLI: history (warm-start episodes), record live episodes, replay them under other speculation sources, report."""
import argparse
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv


def main():
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(prog="fastlane")
    parser.add_argument("command", choices=["history", "record", "replay", "report", "mcp-baseline", "mcp-calibrate",
                                            "mcp-live", "mcp-replay", "mcp-report"])
    parser.add_argument("--config", default=None, help="default: configs/mcp.yaml for mcp-* commands, else configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None, help="history/record: only the first N tasks (smoke run)")
    parser.add_argument("--no-self", action="store_true", help="replay: skip the paid LLM self-prediction arm")
    args = parser.parse_args()
    default = "configs/mcp.yaml" if args.command.startswith("mcp-") else "configs/base.yaml"
    cfg = yaml.safe_load(Path(args.config or default).read_text())
    if args.command == "history":
        from fastlane.runner import history
        history(cfg, args.limit)
    elif args.command == "record":
        from fastlane.runner import record
        record(cfg, args.limit)
    elif args.command == "replay":
        from fastlane.replay import replay_command
        replay_command(cfg, use_self=not args.no_self)
    elif args.command == "report":
        from fastlane.report import write_report
        write_report(cfg)
    elif args.command in ("mcp-baseline", "mcp-live"):
        from fastlane.mcp_runner import mcp_run
        mcp_run(cfg, args.command.removeprefix("mcp-"), args.limit)
    elif args.command == "mcp-calibrate":
        from fastlane.mcp_runner import calibrate_command
        calibrate_command(cfg)
    elif args.command == "mcp-replay":
        from fastlane.replay import mcp_replay_command
        mcp_replay_command(cfg, use_self=not args.no_self)
    elif args.command == "mcp-report":
        from fastlane.report_mcp import write_mcp_report
        write_mcp_report(cfg)


if __name__ == "__main__":
    main()

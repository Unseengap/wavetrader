#!/usr/bin/env python3
"""
WaveTrader Dashboard — Entry Point

Usage:
    python -m dashboard.run                    # default: 0.0.0.0:5000
    python -m dashboard.run --port 8080        # custom port
    python -m dashboard.run --debug            # debug mode with auto-reload
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so wavetrader is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="WaveTrader Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    from dashboard.app import create_app

    app = create_app()

    print(f"\n{'='*60}")
    print(f"  WaveTrader Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Debug: {args.debug}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

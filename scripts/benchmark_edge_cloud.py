from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iedi.edge import NetworkProfile
from iedi.metrics import summarize
from iedi.schemas import InterpretationRequest
from iedi.transport import EdgeInterpretationClient, HttpEdgeCloudTransport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw Paper 4 edge/cloud route and payload observations."
    )
    parser.add_argument("--utterance", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--runs", type=int, default=10)
    audio = parser.add_mutually_exclusive_group(required=True)
    audio.add_argument(
        "--raw-audio-file",
        type=Path,
        help="Measured baseline file; size and SHA-256 are read by this process.",
    )
    audio.add_argument(
        "--raw-audio-bytes",
        type=int,
        help="Hypothetical dry-run size; output is labelled as caller-declared.",
    )
    parser.add_argument("--uplink-kbps", type=float, required=True)
    parser.add_argument("--downlink-kbps", type=float, required=True)
    parser.add_argument("--base-rtt-ms", type=float, required=True)
    parser.add_argument("--network-name", required=True)
    parser.add_argument("--emulator", required=True, help="Descriptive metadata only")
    parser.add_argument(
        "--network-evidence-file",
        type=Path,
        help="Optional captured emulator configuration or cellular test log to hash.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    if args.raw_audio_file is not None:
        if not args.raw_audio_file.is_file():
            raise SystemExit(f"raw audio file not found: {args.raw_audio_file}")
        raw_bytes = args.raw_audio_file.read_bytes()
        if not raw_bytes:
            raise SystemExit("raw audio file is empty")
        raw_audio_bytes = len(raw_bytes)
        raw_audio_source = {
            "kind": "measured_file",
            "path": str(args.raw_audio_file.resolve()),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
    else:
        if args.raw_audio_bytes is None or args.raw_audio_bytes <= 0:
            raise SystemExit("--raw-audio-bytes must be positive")
        raw_audio_bytes = args.raw_audio_bytes
        raw_audio_source = {"kind": "caller_declared_hypothetical", "sha256": None}

    network_evidence_sha256 = None
    if args.network_evidence_file is not None:
        if not args.network_evidence_file.is_file():
            raise SystemExit(
                f"network evidence file not found: {args.network_evidence_file}"
            )
        network_evidence_sha256 = hashlib.sha256(
            args.network_evidence_file.read_bytes()
        ).hexdigest()
    client = EdgeInterpretationClient(HttpEdgeCloudTransport(args.endpoint))
    network = NetworkProfile(
        name=args.network_name,
        uplink_kbps=args.uplink_kbps,
        downlink_kbps=args.downlink_kbps,
        base_rtt_ms=args.base_rtt_ms,
        emulator=args.emulator,
    )

    records = []
    for run_index in range(args.runs):
        observation = client.interpret(
            InterpretationRequest(args.utterance),
            raw_audio_bytes=raw_audio_bytes,
            network_profile=network,
        )
        result = observation.result
        records.append(
            {
                "run": run_index,
                "route": result["decision"]["used_route"],
                "model": result["decision"]["model_id"],
                "ambiguity": result["decision"]["ambiguity_score"],
                "route_reasons": result["decision"]["reasons"],
                "payload_bytes": observation.wire_payload_bytes,
                "raw_audio_bytes": raw_audio_bytes,
                "raw_audio_source": raw_audio_source,
                "payload_reduction": observation.payload_reduction,
                "timing": {
                    "edge_serialization_ms": observation.serialization_ms,
                    "estimated_uplink_transfer_ms": observation.estimated_transfer_ms,
                    "observed_gateway_round_trip_ms": observation.observed_gateway_round_trip_ms,
                    "cloud_api_round_trip_ms": result["timing"]["observed_api_round_trip_ms"],
                    "cloud_handler_end_to_end_ms": result["timing"]["end_to_end_ms"],
                },
                "network": asdict(network),
                "network_evidence_sha256": network_evidence_sha256,
                "network_profile_is_applied_by_this_script": False,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": args.runs,
                "raw_audio_source_kind": raw_audio_source["kind"],
                "network_evidence_sha256": network_evidence_sha256,
                "gateway_round_trip_ms": asdict(
                    summarize(record["timing"]["observed_gateway_round_trip_ms"] for record in records)
                ),
                "cloud_api_round_trip_ms": asdict(
                    summarize(record["timing"]["cloud_api_round_trip_ms"] for record in records)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

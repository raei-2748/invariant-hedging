        ig = None
        if isinstance(diag_record, Mapping):
            ig = (
                diag_record.get("IG", {}).get("ES95")
                if isinstance(diag_record.get("IG"), Mapping)
                else None
            )
        if ig is None:
            ig = _gap(train_vals)

        wg = None
        if isinstance(diag_record, Mapping):
            wg = (
                diag_record.get("WG", {}).get("ES95")
                if isinstance(diag_record.get("WG"), Mapping)
                else None
            )
        if wg is None:
            wg = _compute_wg(train_vals, test_vals)

        msi = None
        if isinstance(diag_record, Mapping):
            msi_obj = diag_record.get("MSI")
            if isinstance(msi_obj, Mapping):
                msi = msi_obj.get("value")

        train_max = _max_or_nan(train_vals)
        train_min = _min_or_nan(train_vals)
        val_high = _extract_single_env_metric(env_metrics, val_envs, "val", "ES95")
        crisis_es = _extract_single_env_metric(env_metrics, test_envs, "test", "ES95")
        crisis_es99 = _extract_single_env_metric(env_metrics, test_envs, "test", "ES99")
        if crisis_es is None:
            crisis_es = _find_metric(final_metrics, split, "es95")
        if crisis_es99 is None:
            crisis_es99 = _find_metric(final_metrics, split, "es99")
        mean_pnl = _find_metric(final_metrics, split, "meanpnl")
        turnover = _find_metric(final_metrics, split, "turnover")

        row = {
            "method": method,
            "seed": int(meta.seed),
            "ig": float(ig) if ig is not None else math.nan,
            "wg": float(wg) if wg is not None else math.nan,
            "msi": float(msi) if msi is not None else math.nan,
            "es95_crisis": float(crisis_es) if crisis_es is not None else math.nan,
            "es99_crisis": float(crisis_es99) if crisis_es99 is not None else math.nan,
            "meanpnl_crisis": float(mean_pnl) if mean_pnl is not None else math.nan,
            "turnover_crisis": float(turnover) if turnover is not None else math.nan,
            "es95_train_max": train_max,
            "es95_train_min": train_min,
            "es95_val_high": float(val_high) if val_high is not None else math.nan,
            "config_tag": meta.config_tag,
        }
        key = (method, int(meta.seed))
        existing = records.get(key)
        if existing is None or combined_mtime >= existing[0]:
            records[key] = (combined_mtime, row)
        if meta.config_tag and method not in config_tags:
            config_tags[method] = meta.config_tag

    resolved = {key: value for key, (mtime, value) in records.items()}
    return resolved, config_tags


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "method",
        "seed",
        "ig",
        "wg",
        "msi",
        "es95_crisis",
        "es99_crisis",
        "meanpnl_crisis",
        "turnover_crisis",
        "es95_train_max",
        "es95_train_min",
        "es95_val_high",
        "commit",
        "phase",
        "config_tag",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    run_roots = _candidate_run_roots(Path("runs"))
    if args.run_roots:
        for token in args.run_roots.split(os.pathsep):
            token = token.strip()
            if token:
                run_roots.append(Path(token))

    LOGGER.info(
        "Scanning diagnostic runs under: %s",
        ", ".join(str(p) for p in run_roots),
    )

    diag_records, config_tags = _collect_diagnostics(
        run_roots,
        args.methods,
        args.seeds,
        args.split,
        args.train_envs,
        args.val_envs,
        args.test_envs,
    )

    rows: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            key = (method, seed)
            base_row = diag_records.get(key, _empty_row(method, seed))
            row = dict(base_row)
            row["commit"] = args.commit_hash
            row["phase"] = args.phase
            if not row.get("config_tag"):
                if method in config_tags:
                    row["config_tag"] = config_tags[method]
                elif args.config_tag:
                    row["config_tag"] = args.config_tag
            rows.append(row)

    rows.sort(key=lambda item: (item["method"], item["seed"]))

    out_path = Path(args.out)
    _ensure_outdir(out_path)
    _write_csv(out_path, rows)
    LOGGER.info("Diagnostics written to %s", out_path)

    meta = {
        "commit": args.commit_hash,
        "phase": args.phase,
        "split": args.split,
        "methods": args.methods,
        "seeds": args.seeds,
        "train_envs": args.train_envs,
        "val_envs": args.val_envs,
        "test_envs": args.test_envs,
        "config_tags": config_tags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    LOGGER.debug("Metadata written to %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

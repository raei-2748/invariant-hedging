377
    parser.add_argument("--methods", required=True, help="Comma-separated list of methods")
378
    parser.add_argument("--seeds", required=True, help="Seed specification, e.g. 0..29")
379
    parser.add_argument("--train_envs", required=True, help="Comma-separated training environments")
380
    parser.add_argument("--val_envs", required=True, help="Comma-separated validation environments")
381
    parser.add_argument("--test_envs", required=True, help="Comma-separated test environments")
382
    parser.add_argument("--out", required=True, help="Output CSV path")
383
    parser.add_argument("--split", default="crisis", help="Test split name for summary metrics")
384
    parser.add_argument("--phase", default="phase2", help="Experiment phase label")
385
    parser.add_argument("--commit_hash", default="UNKNOWN", help="Commit hash for provenance")
386
    parser.add_argument("--config_tag", default=None, help="Optional config tag override")
387
    parser.add_argument(
388
        "--run_roots", default=None, help="Additional run directories (os.pathsep separated)"
389
    )
390
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
391
    args = parser.parse_args(argv)
392
    args.methods = _parse_methods(args.methods)
393
    args.seeds = _parse_range(args.seeds)
394
    args.train_envs = _parse_envs(args.train_envs)
395
    args.val_envs = _parse_envs(args.val_envs)
396
    args.test_envs = _parse_envs(args.test_envs)
397
    return args
398
​
399
​
400
def _collect_diagnostics(
401
    roots: Sequence[Path],
402
    methods: Sequence[str],
403
    seeds: Sequence[int],
404
    split: str,
405
    train_envs: Sequence[str],
406
    val_envs: Sequence[str],
407
    test_envs: Sequence[str],
408
) -> Tuple[Dict[Tuple[str, int], Dict[str, object]], Dict[str, str]]:
409
    method_lookup = {method.upper(): method for method in methods}
410
    seed_set = set(seeds)
411
    records: Dict[Tuple[str, int], Tuple[float, Dict[str, object]]] = {}
412
    config_tags: Dict[str, str] = {}
413
    for meta in _iter_run_metadata(roots):
414
        if meta.method is None or meta.seed is None:
415
            continue
416
        canonical = meta.method.upper()
417
        method = method_lookup.get(canonical)
418
        if method is None:
419
            continue
420
        if meta.seed not in seed_set:
421
            continue
422
        metrics_path = meta.path / "final_metrics.json"
423
        final_metrics = _load_json(metrics_path)
424
        diag_record, diag_mtime = _load_diagnostics_record(meta.path)
425
        metrics_mtime = (
426
            metrics_path.stat().st_mtime if metrics_path.exists() else meta.path.stat().st_mtime
427
        )
428
        combined_mtime = max(metrics_mtime, diag_mtime)
429
        env_metrics = diag_record.get("env_metrics") if isinstance(diag_record, Mapping) else None
430
        train_vals = _collect_env_values(env_metrics, train_envs, "train", "ES95")
431
        test_vals = _collect_env_values(env_metrics, test_envs, "test", "ES95")
432
        ig = None
433
        if isinstance(diag_record, Mapping):
434
            ig = (
435
                diag_record.get("IG", {}).get("ES95")
436
                if isinstance(diag_record.get("IG"), Mapping)
437
                else None
438
            )
439
        if ig is None:
440
            ig = _gap(train_vals)
441
​
442
        wg = None
443
        if isinstance(diag_record, Mapping):
444
            wg = (
445
                diag_record.get("WG", {}).get("ES95")
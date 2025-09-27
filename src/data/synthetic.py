    # Use instantaneous variance from log returns for GBM, or from Heston generator if available.
    params = dynamics_cfg["params"]
    implied_vol = np.zeros_like(spot_paths)
    
    model_name = dynamics_cfg["model"].lower()
    params = dynamics_cfg.get("params", {})
    if model_name == "gbm":
        init_vol = float(params.get("sigma", 0.2))
    elif model_name == "heston":
        init_var = float(params.get("v0", params.get("long_term_var", 0.2)))
        init_vol = math.sqrt(max(init_var, 1e-8))
    else:
        init_vol = 0.2
    implied_vol[:, 0] = max(init_vol, 1e-6)

    dt = getattr(generator, "dt", 1.0 / 252.0)
    inv_sqrt_dt = 1.0 / math.sqrt(max(dt, 1e-12))
    for t in range(1, steps + 1):
        # Cross-sectional realized volatility scaled to annualized units.
        ret = np.log(spot_paths[:, t] / spot_paths[:, t - 1])
        realized = np.std(ret, ddof=0) * inv_sqrt_dt
        implied_vol[:, t] = np.clip(realized, 1e-6, None)

    implied_vol = np.where(np.isfinite(implied_vol), implied_vol, init_vol)

    strike_mode = env_cfg.get("options", {}).get("strike_mode", "atm")
    if strike_mode == "atm":
        strikes = spot_paths[:, [0]]
    else:
        strikes = np.full((num_episodes, 1), env_cfg["options"].get("strike", spot0))

    option_prices = np.zeros_like(spot_paths)
    for t in range(steps + 1):
        tau = np.clip(time_grid[:, t], 1e-6, None)
        sigma_t = np.clip(implied_vol[:, t], 1e-4, None)
        option_prices[:, t] = black_scholes_price(
            spot_paths[:, t], strikes[:, 0], rate_env, sigma_t, tau, option_type="call"
        )

    return EpisodeBatch(
        spot=torch.from_numpy(spot_paths).float(),
        option_price=torch.from_numpy(option_prices).float(),
        implied_vol=torch.from_numpy(implied_vol).float(),
        time_to_maturity=torch.from_numpy(np.ascontiguousarray(time_grid)).float(),
        rate=rate_env,
        env_name=env_name,
        meta={
            "linear_bps": float(costs_cfg.get("linear_bps", 0.0)),
            "quadratic": float(costs_cfg.get("quadratic", 0.0)),
            "slippage_multiplier": float(costs_cfg.get("slippage_multiplier", 1.0)),
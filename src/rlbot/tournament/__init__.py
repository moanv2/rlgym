"""Bot-vs-bot tournament: roster discovery, single-elim bracket, ranking, video capture.

Public, import-light surface (safe to import without torch / rlgym_sim):
    - roster:  ROSTER_CONFIG, Bot, build_roster, resolve_checkpoint
    - bracket: run_bracket, seed_order, final_ranking  (pure logic)
    - match:   tally_games, decide                      (pure logic)

Heavy pieces (load torch / rlgym_sim lazily, only when actually run):
    - match.play_match / play_game
    - obs.make_env
    - policy_io.load_policy

CLI entry points:
    python -m rlbot.tournament.run        # run the bracket, write results JSON
    python -m rlbot.tournament.record     # render a bot's bracket match for video
    python -m rlbot.tournament.download   # fetch teammate checkpoints from manifest
"""

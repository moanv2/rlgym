Drop each teammate's trained checkpoint here so you can 1v1 them.

WHAT TO ASK MARTIN / NACHI FOR:
  One checkpoint FOLDER containing at least:
    PPO_POLICY.pt          (the trained weights)
    BOOK_KEEPING_VARS.json (metadata)
  (the *_OPTIMIZER.pt / VALUE_NET files are optional - not needed to play)

  Their .pt files are gitignored, so a git pull will NOT bring them.
  They must send the folder directly (Drive / Discord / USB / force-add).

WHERE TO PUT IT:
  teammates/martin/<their timestep folder>/PPO_POLICY.pt
  teammates/nachi/<their timestep folder>/PPO_POLICY.pt
  (a folder of folders is fine - the viewer auto-picks the latest timestep)

THEN RUN (open rlviser.exe first):
  python diego-bots/papaya_1v1_viewer.py --orange teammates/martin --episodes 10 --deterministic
  python diego-bots/papaya_1v1_viewer.py --orange teammates/nachi  --episodes 10 --deterministic

Must be AdvancedObs (107-dim) bots - the viewer validates this and errors if not.

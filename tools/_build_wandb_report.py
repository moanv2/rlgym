"""Build + publish Martin's wandb report (9B bot) into the shared team project."""
import wandb

ENT, PROJ, RUNID = "diego08-ie-university", "rlgym-finalproject", "twotz849"
api = wandb.Api()
run = api.run(f"{ENT}/{PROJ}/{RUNID}")
print("run found:", run.name, run.state)
reward, ent, vloss, sps, kl = ("Policy Reward", "Policy Entropy", "Value Function Loss",
                               "Overall Steps per Second", "Mean KL Divergence")
xkey = "Cumulative Timesteps"

try:
    import wandb_workspaces.reports.v2 as wr
except Exception:
    import wandb.apis.reports as wr

panels = []
for metric, title in [(reward, "Policy Reward (learning curve)"), (ent, "Policy Entropy"),
                      (vloss, "Value Function Loss"), (sps, "Steps per Second"), (kl, "Mean KL Divergence")]:
    if metric:
        panels.append(wr.LinePlot(title=title, x=xkey, y=[metric],
                                  smoothing_factor=0.9, smoothing_type="exponential",
                                  smoothing_show_original=False, layout=wr.Layout(w=12, h=8)))

report = wr.Report(
    entity=ENT, project=PROJ,
    title="Martin's Bot | PPO 9B (1v1 Rocket League)",
    description="From-scratch PPO self-play champion, ~9B steps. #1 in both modes of the team tournament.",
    blocks=[
        wr.H1("Martin's Bot | PPO self-play, ~9B steps"),
        wr.MarkdownBlock(
            "**What it is:** PPO actor-critic (1024x3 MLP), AdvancedObs (107-dim) in, one of 90 "
            "LookupAction presets out, ~15 decisions/sec. Trained purely by self-play, kickstarted "
            "(distilled) from Diego's papaya for the first 150M steps then pure RL so it could surpass it.\n\n"
            "**Reward stack** (zero-sum, team_spirit 0): velocity-player-to-ball 0.1, face-ball 0.05, "
            "velocity-ball-to-goal 0.3, event 8.0 (goal +1, concede -1, shot 0.1, demo 0.1), plus "
            "backboard-defense, ball-away-from-own-goal, aerial-touch shaping.\n\n"
            "**Result:** #1 in BOTH modes of the 5-bot tournament (stochastic 73%, deterministic 82%)."
        ),
        wr.H1("Training curves"),
        wr.PanelGrid(runsets=[wr.Runset(entity=ENT, project=PROJ, query=RUNID)], panels=panels),
        wr.MarkdownBlock(
            "**Reading the curves:** Policy Reward rises as it learns to attack/defend/finish. "
            "Policy Entropy falls as the policy sharpens from exploration to decisive play. "
            "Value Function Loss settles as the critic learns to value game states. "
            "SPS ~9k on a Ryzen 9 3900X (this task is CPU-bound on the physics sim, GPU mostly idle)."
        ),
    ],
)
report.save()
print("REPORT URL:", report.url)

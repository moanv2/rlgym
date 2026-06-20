"""Kickstarting-style policy distillation for rlgym-ppo.

Adds an *annealed* teacher->student KL term to the PPO policy loss so our learner is
"kickstarted" toward a stronger teacher bot (Diego's papaya) while still doing its OWN
self-play RL -- so it can SURPASS the teacher, not just imitate it (DeepMind, "Kickstarting
Deep Reinforcement Learning", 2018). Rollouts stay full-speed self-play; the teacher only
runs as a BATCHED forward pass during the PPO update (NO per-step opponent rollout), so this
trains at ~normal self-play speed -- unlike best-response sparring (recipeG, ~6x slower).

Design constraints (critical -- see [[rlgym-working-prefs]] / the champion's safety):
  * NEVER edit the installed rlgym_ppo -- the running champion imports it; a bad edit would
    break its relaunch. Everything here is a SUBCLASS + a method-swap living in our src/.
  * The teacher is FROZEN (eval + requires_grad=False), never added to an optimizer, and
    never written into the checkpoint -- it is re-loaded from its path on every (re)launch.
  * beta == 0  =>  the distill block is skipped  =>  byte-identical to stock PPO. So a
    watchdog relaunch with distillation off is safe, and the equivalence is unit-testable
    (checkpoints/_test_distill.py).
  * We WARM-START from a strong champion at ~2.5B cumulative steps, so the anneal is measured
    RELATIVE to THIS run's starting timestep, not the absolute cumulative count.

Activated by a top-level ``distill:`` block in the training config (see train.py)::

    distill:
      teacher: checkpoints/_eval_snapshots/diego_papaya_1.34B
      obs: advanced          # must match the student's obs (the buffer stores student obs)
      dim: 107
      beta: 0.3              # starting KL weight
      beta_end: 0.0          # annealed to this...
      beta_anneal_steps: 150000000   # ...linearly over this many steps of THIS run
"""

from __future__ import annotations

import os
import re
import time
import types

import numpy as np
import torch
from rlgym_ppo import Learner

from rlbot.evaluation.evaluate import _load_policy, _resolve_checkpoint


def _load_teacher(path: str, obs_dim: int, n_actions: int, device: str):
    """Load papaya (or any LookupAction bot) as a FROZEN teacher policy on ``device``.

    Reuses the same loader the eval harness + FrozenOpponent use, then freezes it so no
    gradients accumulate and it can never be stepped by an optimizer.
    """
    teacher = _load_policy(_resolve_checkpoint(path), obs_dim, n_actions, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def _distill_learn(self, exp):
    """Forked ``PPOLearner.learn`` with an annealed teacher-KL term in the policy loss.

    Copied VERBATIM from rlgym_ppo.ppo.ppo_learner.PPOLearner.learn (pinned version) with
    ONLY the two blocks marked ``# --- DISTILL`` added. If rlgym_ppo is ever upgraded,
    re-sync this body; the beta=0 equivalence test guards against silent copy-drift.
    """
    # --- DISTILL: this run's annealed beta, computed once per update -------------------
    elapsed = self._distill_ts_getter() - self._distill_start_ts
    frac = (
        min(1.0, max(0.0, elapsed / self._distill_anneal_steps))
        if self._distill_anneal_steps > 0
        else 1.0
    )
    beta = self._distill_beta_start + (self._distill_beta_end - self._distill_beta_start) * frac
    mean_distill = 0.0
    # -----------------------------------------------------------------------------------

    n_iterations = 0
    n_minibatch_iterations = 0
    mean_entropy = 0
    mean_divergence = 0
    mean_val_loss = 0
    clip_fractions = []

    # Save parameters before computing any updates.
    policy_before = torch.nn.utils.parameters_to_vector(self.policy.parameters()).cpu()
    critic_before = torch.nn.utils.parameters_to_vector(self.value_net.parameters()).cpu()

    t1 = time.time()
    for _epoch in range(self.n_epochs):
        # Get all shuffled batches from the experience buffer.
        batches = exp.get_all_batches_shuffled(self.batch_size)
        for batch in batches:
            (
                batch_acts,
                batch_old_probs,
                batch_obs,
                batch_target_values,
                batch_advantages,
            ) = batch
            batch_acts = batch_acts.view(self.batch_size, -1)
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                # Send everything to the device and enforce correct shapes.
                start = minibatch_slice
                stop = start + self.mini_batch_size

                acts = batch_acts[start:stop].to(self.device)
                obs = batch_obs[start:stop].to(self.device)
                advantages = batch_advantages[start:stop].to(self.device)
                old_probs = batch_old_probs[start:stop].to(self.device)
                target_values = batch_target_values[start:stop].to(self.device)

                # Compute value estimates.
                vals = self.value_net(obs).view_as(target_values)

                # Get policy log probs & entropy.
                log_probs, entropy = self.policy.get_backprop_data(obs, acts)
                log_probs = log_probs.view_as(old_probs)

                # Compute PPO loss.
                ratio = torch.exp(log_probs - old_probs)
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

                # Compute KL divergence & clip fraction using SB3 method for reporting.
                with torch.no_grad():
                    log_ratio = log_probs - old_probs
                    kl = (torch.exp(log_ratio) - 1) - log_ratio
                    kl = kl.mean().detach().cpu().item()

                    # From the stable-baselines3 implementation of PPO.
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).cpu().item()
                    clip_fractions.append(clip_fraction)

                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                minibatch_ratio = self.mini_batch_size / self.batch_size
                value_loss = self.value_loss_fn(vals, target_values) * minibatch_ratio
                ppo_loss = (policy_loss - entropy * self.ent_coef) * minibatch_ratio

                # --- DISTILL: annealed forward-KL(teacher || student) on these obs -----
                # grad pulls the student TOWARD the teacher (cross-entropy term); the
                # teacher's own entropy is constant w.r.t. the student so it only shifts
                # the reported value, not the gradient. Skipped entirely when beta == 0,
                # which makes this path identical to stock PPO.
                if beta > 0.0:
                    student_probs = self.policy.get_output(obs).view(-1, self.policy.n_actions)
                    student_probs = torch.clamp(student_probs, min=1e-11, max=1.0)
                    with torch.no_grad():
                        teacher_probs = self.teacher.get_output(obs).view(-1, self.teacher.n_actions)
                        teacher_probs = torch.clamp(teacher_probs, min=1e-11, max=1.0)
                    distill_kl = (
                        (teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs)))
                        .sum(dim=-1)
                        .mean()
                    )
                    ppo_loss = ppo_loss + beta * distill_kl * minibatch_ratio
                    mean_distill += distill_kl.detach().cpu().item()
                # -----------------------------------------------------------------------

                ppo_loss.backward()
                value_loss.backward()

                mean_val_loss += (value_loss / minibatch_ratio).cpu().detach().item()
                mean_divergence += kl
                mean_entropy += entropy.cpu().detach().item()
                n_minibatch_iterations += 1

            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            n_iterations += 1

    if n_iterations == 0:
        n_iterations = 1

    if n_minibatch_iterations == 0:
        n_minibatch_iterations = 1

    # Compute averages for the metrics that will be reported.
    mean_entropy /= n_minibatch_iterations
    mean_divergence /= n_minibatch_iterations
    mean_val_loss /= n_minibatch_iterations
    mean_distill /= n_minibatch_iterations  # --- DISTILL
    mean_clip = 0 if len(clip_fractions) == 0 else np.mean(clip_fractions)

    # Compute magnitude of updates made to the policy and value estimator.
    policy_after = torch.nn.utils.parameters_to_vector(self.policy.parameters()).cpu()
    critic_after = torch.nn.utils.parameters_to_vector(self.value_net.parameters()).cpu()
    policy_update_magnitude = (policy_before - policy_after).norm().item()
    critic_update_magnitude = (critic_before - critic_after).norm().item()

    # Assemble and return report dictionary.
    self.cumulative_model_updates += n_iterations

    report = {
        "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
        "Cumulative Model Updates": self.cumulative_model_updates,
        "Policy Entropy": mean_entropy,
        "Mean KL Divergence": mean_divergence,
        "Value Function Loss": mean_val_loss,
        "SB3 Clip Fraction": mean_clip,
        "Policy Update Magnitude": policy_update_magnitude,
        "Value Function Update Magnitude": critic_update_magnitude,
        "Distill Teacher KL": mean_distill,  # --- DISTILL
        "Distill Beta": beta,  # --- DISTILL
    }
    # --- DISTILL: the stock console reporter ignores unknown keys, so surface our two
    # metrics ourselves (once per iteration). teacher_KL trending DOWN = student absorbing
    # the teacher; beta should be annealing toward 0.
    print(
        f"[distill] ts={self._distill_ts_getter():,} beta={beta:.4f} "
        f"teacher_KL={mean_distill:.4f}",
        flush=True,
    )
    self.policy_optimizer.zero_grad()
    self.value_optimizer.zero_grad()

    return report


class DistillLearner(Learner):
    """rlgym-ppo ``Learner`` that kickstarts the policy toward a frozen teacher via an
    annealed KL term. Self-play rollouts run at full speed; the teacher runs only as a
    batched forward pass inside the PPO update. See the module docstring."""

    def __init__(self, *args, distill: dict, **kwargs):
        super().__init__(*args, **kwargs)

        pl = self.ppo_learner
        obs_dim = pl.policy.model[0].in_features  # input layer width == obs size
        n_actions = int(pl.policy.n_actions)
        teacher_path = distill["teacher"]
        teacher_dim = int(distill.get("dim", obs_dim))

        # The teacher consumes the SAME obs vectors stored in the student's buffer, so its
        # obs builder/dim MUST match the student's. (papaya = advanced 107 == our advanced 107.)
        assert teacher_dim == obs_dim, (
            f"distill teacher obs dim ({teacher_dim}) must equal the student's ({obs_dim}); "
            "the buffer stores the student's obs and feeds them straight to the teacher."
        )

        pl.teacher = _load_teacher(teacher_path, teacher_dim, n_actions, self.device)
        pl._distill_beta_start = float(distill.get("beta", 0.3))
        pl._distill_beta_end = float(distill.get("beta_end", 0.0))
        pl._distill_anneal_steps = float(distill.get("beta_anneal_steps", 150_000_000))
        # Anneal RELATIVE to where THIS distill run first began, PERSISTED across watchdog
        # relaunches. Without persistence, every relaunch would recapture cumulative_timesteps
        # (already ~2.5B from the warm-start) as the start -> elapsed resets to 0 -> beta jumps
        # back to beta_start on every restart and never anneals to 0. So we stamp the start ts
        # once into a sidecar keyed by experiment name (stable; the save folder gets a fresh
        # unix suffix each launch) and read it back on resume.
        pl._distill_start_ts = self._distill_load_or_stamp_start_ts()
        pl._distill_ts_getter = lambda: self.agent.cumulative_timesteps

        pl.learn = types.MethodType(_distill_learn, pl)

        print(
            f"[distill] teacher={teacher_path} (obs_dim={teacher_dim}, {n_actions} actions) | "
            f"beta {pl._distill_beta_start} -> {pl._distill_beta_end} over "
            f"{pl._distill_anneal_steps:,.0f} steps, anneal start ts {pl._distill_start_ts:,} "
            f"(now at {self.agent.cumulative_timesteps:,})",
            flush=True,
        )

    def _distill_load_or_stamp_start_ts(self) -> int:
        """Return this distill run's anneal-origin timestep, persisted across relaunches.

        First launch stamps the current cumulative_timesteps; every resume reads it back.
        Keyed by the experiment base name: strip ONLY the trailing ``-<unix>`` suffix that
        Learner appends to the save folder (regex-anchored to the end, so hyphens elsewhere
        in the path or experiment name are safe).
        """
        base = re.sub(r"-\d+$", "", self.checkpoints_save_folder)
        start_file = base + "_diststart.txt"
        if os.path.exists(start_file):
            with open(start_file, encoding="utf-8") as f:
                return int(f.read().strip())
        start_ts = int(self.agent.cumulative_timesteps)
        os.makedirs(os.path.dirname(start_file), exist_ok=True)
        with open(start_file, "w", encoding="utf-8") as f:
            f.write(str(start_ts))
        return start_ts

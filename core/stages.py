"""A tiny stage runner, so one script covers one research question.

The playbook has ~30 experiments across RQ1-RQ7. One script each would be ~30
entry points that mostly differ in which cached artifact they read, and the
expensive step — activation capture — would be repeated or duplicated between
them.

Instead: **one script per research question**, composed of named stages.

* Stages declare what they `produce`, so a completed stage is **skipped** unless
  `--force`. Re-running an analysis never re-runs a capture.
* Stages declare what they `require`, so the runner resolves order and refuses to
  run a stage whose input is missing.
* Stages declare `needs_gpu`, so a CPU-only analysis pass is obvious and can run
  off the batch queue.
* Stages hand results to each other in memory within one invocation via
  `ctx.shared`, and on disk across invocations via their artifacts.

Per-model stages run once per configured model and write to
`results/<experiment>/<model_slug>/`; model-independent stages write to
`results/<experiment>/`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class Context:
    """What a stage is given."""
    cfg: object
    log: object
    model: Optional[str]        # model slug, or None for model-independent stages
    out: Path                   # this stage's output directory
    shared: Dict[str, object] = field(default_factory=dict)

    def spec(self):
        return self.cfg.spec(self.model)


@dataclass
class Stage:
    name: str
    fn: Callable[[Context], None]
    produces: Sequence[str] = ()
    requires: Sequence[str] = ()
    needs_gpu: bool = False
    per_model: bool = True
    doc: str = ""

    def complete(self, out: Path) -> bool:
        return bool(self.produces) and all((out / f).exists() for f in self.produces)


class Pipeline:
    def __init__(self, name: str, stages: Sequence[Stage]):
        self.name = name
        self.stages = list(stages)
        self.by_name = {s.name: s for s in self.stages}
        for s in self.stages:                       # fail fast on a typo in `requires`
            for r in s.requires:
                if r not in self.by_name:
                    raise ValueError(f"stage {s.name!r} requires unknown stage {r!r}")

    # -- selection ---------------------------------------------------------
    def _select(self, only: Optional[List[str]], start: Optional[str]) -> List[Stage]:
        if only:
            unknown = [o for o in only if o not in self.by_name]
            if unknown:
                raise ValueError(f"unknown stage(s) {unknown}; known: {list(self.by_name)}")
            return [s for s in self.stages if s.name in set(only)]
        if start:
            if start not in self.by_name:
                raise ValueError(f"unknown stage {start!r}")
            i = [s.name for s in self.stages].index(start)
            return self.stages[i:]
        return list(self.stages)

    # -- execution ---------------------------------------------------------
    def run(self, cfg, log, only=None, start=None, force=False, dry_run=False) -> None:
        selected = self._select(only, start)
        log.info(f"pipeline {self.name}: {[s.name for s in selected]}"
                 + (" (dry run)" if dry_run else ""))

        shared: Dict[str, Dict[str, object]] = {}
        for stage in selected:
            models = cfg.models if stage.per_model else [None]
            for model in models:
                out = cfg.dir(self.name, model) if model else cfg.dir(self.name)
                tag = f"{stage.name}" + (f"/{model}" if model else "")

                missing = [r for r in stage.requires
                           if not self.by_name[r].complete(
                               cfg.dir(self.name, model) if self.by_name[r].per_model and model
                               else cfg.dir(self.name))]
                if missing and not dry_run:
                    log.error(f"[{tag}] SKIPPED — required stage(s) incomplete: {missing}")
                    continue

                if stage.complete(out) and not force:
                    log.info(f"[{tag}] already complete ({len(stage.produces)} artifacts) — skipping")
                    continue
                if dry_run:
                    log.info(f"[{tag}] would run{' (GPU)' if stage.needs_gpu else ''}")
                    continue

                log.info("-" * 78)
                log.info(f"[{tag}] {stage.doc or stage.name}")
                ctx = Context(cfg=cfg, log=log, model=model, out=out,
                              shared=shared.setdefault(model or "_", {}))
                stage.fn(ctx)


def add_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--only", nargs="*", help="run only these stages")
    parser.add_argument("--from", dest="start", help="run from this stage onwards")
    parser.add_argument("--force", action="store_true", help="re-run even if artifacts exist")
    parser.add_argument("--dry-run", action="store_true", help="show what would run")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    return parser


def maybe_list(pipeline: Pipeline, args) -> bool:
    if not getattr(args, "list", False):
        return False
    print(f"{pipeline.name} stages:")
    for s in pipeline.stages:
        print(f"  {s.name:<16} {'GPU' if s.needs_gpu else 'cpu'}  "
              f"{'per-model' if s.per_model else 'shared   '}  "
              f"requires={list(s.requires) or '-'}")
        if s.doc:
            print(f"    {s.doc}")
    return True

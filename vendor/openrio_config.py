"""YAML configuration overlay for OpenRiO-only registered components."""

from __future__ import annotations

from engine.core import YAMLConfig as UpstreamYAMLConfig
from engine.core.workspace import create


class OpenRiOYAMLConfig(UpstreamYAMLConfig):
    """Extend, without editing, the upstream evaluator type allow-list."""

    @property
    def evaluator(self):
        evaluator_cfg = self.yaml_cfg.get("evaluator", {})
        if evaluator_cfg.get("type") != "OpenRiODotaEvaluator":
            return super().evaluator
        if self._evaluator is None:
            self._evaluator = create(
                "evaluator",
                self.global_cfg,
                dataset=self.val_dataloader.dataset,
            )
        return self._evaluator


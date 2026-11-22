# Copyright 2026 Kilian Ludwig
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from utils.parameters import count_parameters


@dataclass
class OutputConfig:
    
    dir: str = "./experiments"
    run_name: Optional[str] = None
    save_checkpoint: bool = True
    save_metrics: bool = True
    save_config: bool = True
    
    checkpoint_save_last: bool = True
    checkpoint_save_best: bool = True
    checkpoint_interval: Optional[int] = None
    
    plots_enabled: bool = True
    plots_format: str = "pdf"
    plots_dpi: int = 300
    plots_training_curves: bool = True
    plots_include_val: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        plots = data.get("plots", {})
        interval = data.get("checkpoint_interval")
        if interval is not None:
            interval = int(interval)
        return cls(
            dir=data.get("dir", "./experiments"),
            run_name=data.get("run_name"),
            save_checkpoint=data.get("save_checkpoint", True),
            save_metrics=data.get("save_metrics", True),
            save_config=data.get("save_config", True),
            checkpoint_save_last=data.get("checkpoint_save_last", True),
            checkpoint_save_best=data.get("checkpoint_save_best", True),
            checkpoint_interval=interval,
            plots_enabled=plots.get("enabled", True),
            plots_format=plots.get("format", "pdf"),
            plots_dpi=plots.get("dpi", 300),
            plots_training_curves=plots.get("training_curves", True),
            plots_include_val=plots.get("include_val", True),
        )


@dataclass
class RunOutput:
    
    output_dir: Path
    run_name: str
    config: Dict[str, Any]
    output_config: OutputConfig = field(default_factory=OutputConfig)
    
    _start_time: float = field(default_factory=time.time, repr=False)
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        if self.output_config.plots_enabled:
            self.plots_dir = self.run_dir / "plots"
            self.plots_dir.mkdir(exist_ok=True)
        
        if self.output_config.save_checkpoint:
            self.checkpoints_dir = self.run_dir / "checkpoints"
            self.checkpoints_dir.mkdir(exist_ok=True)
        else:
            self.checkpoints_dir = None
    
    @classmethod
    def create(
        cls,
        base_dir: str,
        task_name: str,
        model_name: str,
        config: Dict[str, Any],
        output_config: Optional[OutputConfig] = None,
        run_name: Optional[str] = None,
    ) -> "RunOutput":
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_name}_{task_name}_{timestamp}"
        
        return cls(
            output_dir=Path(base_dir),
            run_name=run_name,
            config=config,
            output_config=output_config or OutputConfig(),
        )
    
    def save_config(self) -> Path:
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        return config_path
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        path: Optional[Path] = None,
    ) -> Path:
        if path is None:
            path = (
                self.checkpoints_dir / "last.pt"
                if self.checkpoints_dir is not None
                else self.run_dir / "checkpoint.pt"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": self.config,
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if history is not None:
            checkpoint["history"] = history
        if metadata is not None:
            checkpoint["metadata"] = metadata
        
        checkpoint["parameter_count"] = count_parameters(model).trainable
        
        torch.save(checkpoint, path)
        return path
    
    def save_metrics(
        self,
        history: List[Dict],
        final_metrics: Optional[Dict] = None,
        scaling: Optional[Dict[str, Any]] = None,
    ) -> Path:
        metrics_path = self.run_dir / "metrics.json"
        seed = (
            self.config.get("training", {}).get("seed")
            or self.config.get("task", {}).get("seed")
            or 42
        )
        metrics_data = {
            "history": history,
            "epochs": len(history),
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
        }
        
        if final_metrics:
            metrics_data["final"] = final_metrics
        
        if history:
            best_loss = min((h.get("loss", float("inf")) for h in history), default=float("inf"))
            best_val_loss = min((h.get("val_loss", float("inf")) for h in history if "val_loss" in h), default=float("inf"))
            best_accuracy = max((h.get("accuracy", 0) for h in history if "accuracy" in h), default=0)
            _val_acc = (
                h.get("val/accuracy", h.get("val_accuracy", 0))
                for h in history
                if "val/accuracy" in h or "val_accuracy" in h
            )
            best_val_accuracy = max(_val_acc, default=0)

            best_token_accuracy = max(
                (h.get("token_accuracy", 0) for h in history if "token_accuracy" in h),
                default=0,
            )
            _val_tok_acc = (
                h.get("val/token_accuracy", h.get("val_token_accuracy", 0))
                for h in history
                if "val/token_accuracy" in h or "val_token_accuracy" in h
            )
            best_val_token_accuracy = max(_val_tok_acc, default=0)
            
            metrics_data["best"] = {
                "loss": best_loss if best_loss != float("inf") else None,
                "val_loss": best_val_loss if best_val_loss != float("inf") else None,
                "accuracy": best_accuracy if best_accuracy > 0 else None,
                "val_accuracy": best_val_accuracy if best_val_accuracy > 0 else None,
                "token_accuracy": best_token_accuracy if best_token_accuracy > 0 else None,
                "val_token_accuracy": best_val_token_accuracy if best_val_token_accuracy > 0 else None,
            }
            if final_metrics and final_metrics.get("test_exhausted"):
                metrics_data["best"]["test_exhausted"] = True
                metrics_data["best"]["test_count"] = final_metrics.get("test_count", 0)
                metrics_data["best"]["test_seed"] = final_metrics.get("test_seed")
            elif final_metrics and final_metrics.get("test_accuracy") is not None:
                metrics_data["best"]["test_accuracy"] = final_metrics["test_accuracy"]
                metrics_data["best"]["test_token_accuracy"] = final_metrics.get(
                    "test_token_accuracy"
                )
                metrics_data["best"]["test_count"] = final_metrics.get("test_count")
                metrics_data["best"]["test_seed"] = final_metrics.get("test_seed")
                metrics_data["best"]["best_val_epoch"] = final_metrics.get("best_val_epoch")

        if scaling is not None:
            metrics_data["scaling"] = scaling
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        return metrics_path
    
    def save_run_info(
        self,
        model: torch.nn.Module,
        device: str,
        training_time: Optional[float] = None,
        extra_metadata: Optional[Dict] = None,
        history: Optional[List[Dict]] = None,
    ) -> Path:
        run_info_path = self.run_dir / "run_info.json"
        
        stats = count_parameters(model)
        
        if training_time is None:
            training_time = time.time() - self._start_time
        
        seed = (
            self.config.get("training", {}).get("seed")
            or self.config.get("task", {}).get("seed")
            or 42
        )
        run_info = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "parameters": {
                "trainable": stats.trainable,
                "total": stats.total,
            },
            "training_time_seconds": round(training_time, 2),
            "task": self.config.get("task", {}).get("name"),
            "model": self.config.get("model", {}).get("name"),
            "seed": seed,
        }

        run_info.update(self._timing_aggregates(history))

        sweep_info_path = self.output_dir / "sweep_info.json"
        if sweep_info_path.exists():
            try:
                with open(sweep_info_path) as f:
                    sweep_info = json.load(f)
                run_info["sweep_group"] = str(self.output_dir)
                if "sweep_id" in sweep_info:
                    run_info["sweep_id"] = sweep_info["sweep_id"]
                if "config_id" in sweep_info:
                    run_info["config_id"] = sweep_info["config_id"]
            except (OSError, json.JSONDecodeError):
                pass
        
        if extra_metadata:
            run_info["extra"] = extra_metadata
        
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)
        
        return run_info_path

    @staticmethod
    def _timing_aggregates(history: Optional[List[Dict]]) -> Dict[str, Any]:
        epochs = [h for h in (history or []) if "time/epoch_seconds" in h]
        if not epochs:
            return {}

        def _total(key: str) -> float:
            return sum(float(h.get(key, 0.0)) for h in epochs)

        epoch_total = _total("time/epoch_seconds")
        return {
            "epochs_timed": len(epochs),
            "mean_epoch_seconds": round(epoch_total / len(epochs), 4),
            "train_seconds_total": round(_total("time/train_seconds"), 2),
            "eval_seconds_total": round(_total("time/eval_seconds"), 2),
        }

    def save_plots(self, history: List[Dict]) -> List[Path]:
        if not self.output_config.plots_enabled:
            return []
        
        saved_paths = []
        
        try:
            from visualization import plot_training_curves
            
            if self.output_config.plots_training_curves and history:
                curves_path = self.plots_dir / f"training_curves.{self.output_config.plots_format}"
                plot_training_curves(
                    history=history,
                    output_path=curves_path,
                    include_val=self.output_config.plots_include_val,
                    dpi=self.output_config.plots_dpi,
                )
                saved_paths.append(curves_path)
                
                loss_path = self.plots_dir / f"loss_curve.{self.output_config.plots_format}"
                plot_training_curves(
                    history=history,
                    output_path=loss_path,
                    metrics=["loss", "val_loss"] if self.output_config.plots_include_val else ["loss"],
                    dpi=self.output_config.plots_dpi,
                )
                saved_paths.append(loss_path)
                
                if any("accuracy" in h or "val/accuracy" in h for h in history):
                    acc_metrics = ["accuracy"]
                    if self.output_config.plots_include_val:
                        has_val_slash = any("val/accuracy" in h for h in history)
                        acc_metrics.append("val/accuracy" if has_val_slash else "val_accuracy")
                    acc_path = self.plots_dir / f"accuracy_curve.{self.output_config.plots_format}"
                    plot_training_curves(
                        history=history,
                        output_path=acc_path,
                        metrics=acc_metrics,
                        dpi=self.output_config.plots_dpi,
                    )
                    saved_paths.append(acc_path)

                if any("token_accuracy" in h or "val/token_accuracy" in h for h in history):
                    tok_acc_metrics = ["token_accuracy"]
                    if self.output_config.plots_include_val:
                        has_val_tok_slash = any("val/token_accuracy" in h for h in history)
                        tok_acc_metrics.append("val/token_accuracy" if has_val_tok_slash else "val_token_accuracy")
                    tok_acc_path = self.plots_dir / f"token_accuracy_curve.{self.output_config.plots_format}"
                    plot_training_curves(
                        history=history,
                        output_path=tok_acc_path,
                        metrics=tok_acc_metrics,
                        dpi=self.output_config.plots_dpi,
                    )
                    saved_paths.append(tok_acc_path)
        
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
        
        return saved_paths
    
    def finalize(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: List[Dict],
        device: str,
        training_time: Optional[float] = None,
        extra_metadata: Optional[Dict] = None,
        scaling: Optional[Dict[str, Any]] = None,
        final_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        saved: Dict[str, Any] = {}
        
        if self.output_config.save_config:
            saved["config"] = self.save_config()
        
        if self.output_config.save_checkpoint and self.output_config.checkpoint_save_last:
            if self.checkpoints_dir is None:
                self.checkpoints_dir = self.run_dir / "checkpoints"
                self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            saved["checkpoint"] = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=len(history),
                history=history,
                metadata=extra_metadata,
                path=self.checkpoints_dir / "last.pt",
            )
        
        if self.output_config.save_metrics:
            saved["metrics"] = self.save_metrics(
                history, final_metrics=final_metrics, scaling=scaling,
            )
        
        saved["run_info"] = self.save_run_info(
            model=model,
            device=device,
            training_time=training_time,
            extra_metadata=extra_metadata,
            history=history,
        )
        
        plot_paths = self.save_plots(history)
        if plot_paths:
            saved["plots"] = plot_paths
        
        if self.checkpoints_dir is not None and self.checkpoints_dir.exists():
            saved["checkpoints"] = sorted(self.checkpoints_dir.glob("*.pt"))
        
        return saved


def generate_run_name(model_name: str, task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{task_name}_{timestamp}"


class CheckpointListener:

    def __init__(
        self,
        run_output: "RunOutput",
        output_config: OutputConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.run_output = run_output
        self.output_config = output_config
        self.model = model
        self.optimizer = optimizer
        self.extra_metadata = extra_metadata or {}
        self.history_so_far: List[Dict[str, Any]] = []
        self.best_accuracy: float = 0.0
        self.best_epoch: Optional[int] = None

    def on_train_start(self) -> None:
        self.history_so_far = []
        self.best_accuracy = 0.0
        self.best_epoch = None

    def on_train_end(self, history: List[Dict[str, float]]) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if not self.output_config.save_checkpoint or self.run_output.checkpoints_dir is None:
            return
        self.history_so_far.append(dict(metrics))
        acc = metrics.get("val/accuracy", metrics.get("val_accuracy"))
        if acc is None:
            acc = metrics.get("accuracy", 0.0)
        if acc is None:
            acc = 0.0

        if self.output_config.checkpoint_save_best and acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_epoch = epoch
            self.run_output.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                history=list(self.history_so_far),
                metadata={**self.extra_metadata, "checkpoint_type": "best", "best_accuracy": acc},
                path=self.run_output.checkpoints_dir / "best.pt",
            )

        interval = self.output_config.checkpoint_interval
        if interval is not None and epoch % interval == 0:
            self.run_output.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                history=list(self.history_so_far),
                metadata={**self.extra_metadata, "checkpoint_type": "interval"},
                path=self.run_output.checkpoints_dir / f"epoch_{epoch:05d}.pt",
            )

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

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from ml.models import BaseSequenceModel
from .config import WandbConfig

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

_force_color = bool(os.environ.get("FORCE_COLOR"))
_console = Console(force_terminal=_force_color) if RICH_AVAILABLE else None


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class TrainingProgress:

    def __init__(self, total_epochs: int, use_colors: bool = True, sweep_info: Optional[Dict[str, int]] = None):
        self.total_epochs = total_epochs
        self.sweep_info = sweep_info
        self.start_time: Optional[float] = None
        self.epoch_times: List[float] = []
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self._use_rich = RICH_AVAILABLE and (sys.stdout.isatty() or _force_color)
        self._batch_progress: Optional[Any] = None
        self._batch_task_id: Optional[Any] = None

    def start(self) -> None:
        self.start_time = time.time()

    def get_epoch_pbar(self, dataloader: Iterable, epoch: int):
        total = len(dataloader) if hasattr(dataloader, '__len__') else None

        if self._use_rich:
            self._batch_progress = Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]Epoch {epoch}/{self.total_epochs}[/cyan]"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[loss_str]}"),
                console=_console,
                transient=True,
            )
            self._batch_task_id = self._batch_progress.add_task(
                "training", total=total, loss_str=""
            )
            self._batch_progress.start()
            return _RichBatchIterator(dataloader, self._batch_progress, self._batch_task_id)
        else:
            from tqdm import tqdm
            return tqdm(
                dataloader, total=total,
                desc=f"Epoch {epoch}/{self.total_epochs}",
                unit="batch", ncols=100, leave=False,
            )

    def update_batch(self, pbar, loss: float) -> None:
        if isinstance(pbar, _RichBatchIterator):
            self._batch_progress.update(
                self._batch_task_id, advance=0,
                loss_str=f"loss: {loss:.4f}",
            )
        else:
            pbar.set_postfix({'loss': f'{loss:.4f}'}, refresh=True)

    def end_epoch(self, epoch: int, epoch_time: float, metrics: Dict[str, float]) -> None:
        self.epoch_times.append(epoch_time)

        if self._batch_progress is not None:
            self._batch_progress.stop()
            self._batch_progress = None

        acc = metrics.get('accuracy', 0)
        loss = metrics.get('loss', float('inf'))
        val_loss = metrics.get('val_loss')

        acc_indicator = " [green]★[/green]" if acc > self.best_accuracy else "  "
        loss_indicator = " [green]↓[/green]" if loss < self.best_loss else "  "

        if acc > self.best_accuracy:
            self.best_accuracy = acc
        if loss < self.best_loss:
            self.best_loss = loss

        val_loss_str = ""
        if val_loss is not None:
            vl_ind = " [green]↓[/green]" if val_loss < self.best_val_loss else " [red]↑[/red]"
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            val_loss_str = f"  val_loss: [bright_magenta]{val_loss:.4f}[/bright_magenta]{vl_ind}"

        lr = metrics.get("lr")
        lr_str = f"  lr: [dim]{lr:.2e}[/dim]" if lr is not None else ""

        acc_color = "green" if acc > 0.8 else "yellow" if acc > 0.5 else "red"

        sweep_str = ""
        if self.sweep_info:
            tw = len(str(self.sweep_info['total']))
            sweep_str = f"[cyan]Model {self.sweep_info['current']:>{tw}}/{self.sweep_info['total']}[/cyan]  "

        progress_frac = epoch / self.total_epochs
        bar_w = 20
        filled = int(bar_w * progress_frac)
        bar = f"[bright_blue]{'█' * filled}[/bright_blue][dim]{'░' * (bar_w - filled)}[/dim]"

        line = (
            f"  {sweep_str}[bold]Epoch {epoch:>3}/{self.total_epochs}[/bold] {bar} "
            f"[dim]\\[{_format_time(epoch_time):>7}][/dim]  "
            f"loss: [yellow]{loss:.4f}[/yellow]{loss_indicator}"
            f"{val_loss_str}{lr_str}"
            f"  acc: [{acc_color}]{acc:.2%}[/{acc_color}]{acc_indicator}"
        )

        if self._use_rich and _console:
            _console.print(line, highlight=False)
        else:
            print(line)

    def finish(self, history: List[Dict[str, float]]) -> None:
        if not self.start_time:
            return

        total_time = time.time() - self.start_time

        if not self._use_rich or not _console:
            self._finish_plain(history, total_time)
            return

        lines = []
        if history:
            final = history[-1]
            lines.append("[bold]Final Metrics[/bold]")
            lines.append(f"  Loss:     [yellow]{final.get('loss', 0):.4f}[/yellow]")
            if 'accuracy' in final:
                acc = final['accuracy']
                ac = "green" if acc > 0.9 else "yellow" if acc > 0.7 else "red"
                lines.append(f"  Accuracy: [{ac}]{acc:.2%}[/{ac}]")
            if 'val_loss' in final:
                lines.append(f"  Val Loss: [bright_magenta]{final['val_loss']:.4f}[/bright_magenta]")

        lines.append("")
        lines.append("[bold]Best Results[/bold]")
        lines.append(f"  Best Loss:     [yellow]{self.best_loss:.4f}[/yellow]")
        if self.best_accuracy > 0:
            lines.append(f"  Best Accuracy: [green]{self.best_accuracy:.2%}[/green]")

        lines.append("")
        lines.append("[bold]Timing[/bold]")
        lines.append(f"  Total Time:    {_format_time(total_time)}")
        if self.epoch_times:
            avg = sum(self.epoch_times) / len(self.epoch_times)
            lines.append(f"  Avg per Epoch: {_format_time(avg)}")

        _console.print()
        _console.print(Panel(
            "\n".join(lines),
            title="[bold green]Training Complete[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        _console.print()

    def _finish_plain(self, history: List[Dict[str, float]], total_time: float) -> None:
        print()
        print("─" * 70)
        print("  Training Complete!")
        print("─" * 70)
        if history:
            final = history[-1]
            print(f"  Final loss: {final.get('loss', 0):.4f}")
            if 'accuracy' in final:
                print(f"  Accuracy:   {final['accuracy']:.2%}")
        print(f"  Best loss:     {self.best_loss:.4f}")
        if self.best_accuracy > 0:
            print(f"  Best accuracy: {self.best_accuracy:.2%}")
        print(f"  Total time:    {_format_time(total_time)}")
        print("─" * 70)
        print()


class _RichBatchIterator:

    def __init__(self, dataloader, progress, task_id):
        self._dataloader = dataloader
        self._progress = progress
        self._task_id = task_id

    def __iter__(self):
        for batch in self._dataloader:
            yield batch
            self._progress.update(self._task_id, advance=1)

    def __len__(self):
        return len(self._dataloader) if hasattr(self._dataloader, '__len__') else 0

    def close(self):
        pass


Batch = Dict[str, torch.Tensor]


class TrainingListener:
    """Base class for training event listeners."""

    def on_train_start(self) -> None:
        pass

    def on_train_end(self, history: List[Dict[str, float]]) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass


@dataclass
class TrainerConfig:
    
    device: str = "cpu"
    epochs: int = 10
    log_every: int = 10
    early_stop_at_accuracy: Optional[float] = 0.95
    grad_clip_max_norm: Optional[float] = None
    wandb_config: Optional[WandbConfig] = None

    def __post_init__(self):
        if self.wandb_config is None:
            self.wandb_config = WandbConfig(enabled=False)


class Trainer:

    def __init__(
        self,
        model: BaseSequenceModel,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig,
        scheduler: Optional[LRScheduler] = None,
        listeners: Optional[List[TrainingListener]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        sweep_info: Optional[Dict[str, int]] = None,
    ) -> None:
        self._device = torch.device(config.device) if isinstance(config.device, str) else config.device
        try:
            self.model = model.to(self._device)
        except RuntimeError as exc:
            if "out of memory" in str(exc) and self._device.type == "cuda":
                try:
                    free_mb = torch.cuda.mem_get_info(self._device)[0] / (1024 ** 2)
                    total_mb = torch.cuda.mem_get_info(self._device)[1] / (1024 ** 2)
                    mem_info = f" ({free_mb:.0f}/{total_mb:.0f} MB free)"
                except Exception:
                    mem_info = ""
                raise RuntimeError(
                    f"CUDA out of memory when moving model to {self._device}{mem_info}. "
                    f"Check `nvidia-smi` for other processes using the GPU, or set device=cpu."
                ) from exc
            raise
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.listeners = listeners or []
        self.experiment_metadata = experiment_metadata or {}
        self.wandb_run = None
        self.verbose = verbose
        self.sweep_info = sweep_info
        self.progress: Optional[TrainingProgress] = None
        self._best_val_accuracy = -1.0
        self._best_val_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_epoch: Optional[int] = None

        if self.config.wandb_config.enabled and WANDB_AVAILABLE:
            self._init_wandb()

    def _init_wandb(self) -> None:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available.")
            self.config.wandb_config.enabled = False
            return

        wconfig = self.config.wandb_config

        try:
            if not wandb.api.api_key:
                print("Warning: wandb API key not configured.")
                self.config.wandb_config.enabled = False
                return

            wandb_config = {
                "device": str(self._device),
                "epochs": self.config.epochs,
                "log_every": self.config.log_every,
                **self.experiment_metadata,
            }

            self.wandb_run = wandb.init(
                project=wconfig.project,
                entity=wconfig.entity,
                name=wconfig.run_name,
                tags=wconfig.tags,
                notes=wconfig.notes,
                config=wandb_config,
                reinit=True,
            )

            try:
                wandb.watch(
                    self.model,
                    log="all" if wconfig.log_gradients else "parameters",
                    log_freq=wconfig.log_freq,
                )
            except Exception as e:
                print(f"Warning: Could not set up wandb.watch: {e}")

        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.config.wandb_config.enabled = False
            self.wandb_run = None

    def _log_to_wandb(
        self, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        if self.config.wandb_config.enabled and WANDB_AVAILABLE and self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

    def _finish_wandb(self) -> None:
        if self.config.wandb_config.enabled and WANDB_AVAILABLE and self.wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")

    def _raise_with_context(
        self,
        exc: RuntimeError,
        *,
        epoch: Optional[int],
        step: int,
        phase: str,
    ) -> None:
        msg = str(exc).lower()
        is_oom = "out of memory" in msg or "cuda out of memory" in msg
        is_no_kernel = "no kernel image" in msg

        mem_part = ""
        if self._device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(self._device)
                mem_part = (
                    f"  GPU memory: {free / 1024**2:.0f} / "
                    f"{total / 1024**2:.0f} MB free"
                )
            except Exception:
                mem_part = ""
            try:
                if is_oom:
                    torch.cuda.empty_cache()
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass

        where = (
            f"epoch {epoch} step {step} ({phase})"
            if epoch is not None else f"{phase} step {step}"
        )

        if is_oom:
            hint = (
                "Hint: lower training.batch_size, reduce model size "
                "(d_model/num_layers), or pick a GPU with more memory."
            )
        elif is_no_kernel:
            hint = (
                "Hint: the active GPU's compute capability is not in the "
                "installed wheel's build list. Use a different GPU or "
                "reinstall with the matching arch."
            )
        else:
            hint = "Hint: re-run with CUDA_LAUNCH_BLOCKING=1 to localise the failing op."

        wrapped = RuntimeError(
            f"{exc.__class__.__name__} during {where}: {exc}{mem_part}\n{hint}"
        )
        raise wrapped from exc

    def train_epoch(self, dataloader: Iterable[Batch], epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step_count = 0
        
        if self.progress and self.verbose:
            pbar = self.progress.get_epoch_pbar(dataloader, epoch)
            iterator = pbar
        else:
            pbar = None
            iterator = dataloader
        
        for step_count, batch in enumerate(iterator, start=1):
            try:
                inputs = batch["inputs"].to(self._device)
                targets = batch["targets"].to(self._device)
                outputs, _ = self.model(inputs, targets=targets)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                if self.config.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.grad_clip_max_norm,
                    )
                if self._device.type == "cuda":
                    torch.cuda.synchronize()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
            except RuntimeError as exc:
                self._raise_with_context(
                    exc, epoch=epoch, step=step_count, phase="train",
                )
            
            if pbar is not None:
                self.progress.update_batch(pbar, loss.item())
            
            if step_count % self.config.log_every == 0:
                self._log_to_wandb({
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/step": step_count,
                })
        
        if pbar is not None:
            pbar.close()

        if step_count == 0:
            return 0.0
        return total_loss / step_count

    def evaluate(self, dataloader: Iterable[Batch]) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        exact_seq_correct = 0
        token_correct = 0
        total_samples = 0
        total_tokens = 0

        with torch.no_grad():
            for step_idx, batch in enumerate(dataloader, start=1):
                try:
                    inputs = batch["inputs"].to(self._device)
                    targets = batch["targets"].to(self._device)
                    outputs, _ = self.model(inputs)

                    loss = self.loss_fn(outputs, targets)
                    if self._device.type == "cuda":
                        torch.cuda.synchronize()
                    total_loss += loss.item()

                    if targets.dim() == 1:
                        targets_2d = targets.unsqueeze(1)
                    else:
                        targets_2d = targets

                    if outputs.dim() == 2:
                        outputs_3d = outputs.unsqueeze(1)
                    else:
                        outputs_3d = outputs

                    seq_len = targets_2d.size(1)
                    preds = torch.argmax(outputs_3d[:, :seq_len, :], dim=-1)

                    mask = targets_2d != -100
                    matches = (preds == targets_2d) & mask

                    exact_seq_correct += ((preds == targets_2d) | ~mask).all(dim=-1).sum().item()
                    total_samples += targets_2d.size(0)

                    token_correct += matches.sum().item()
                    total_tokens += mask.sum().item()
                except RuntimeError as exc:
                    self._raise_with_context(
                        exc, epoch=None, step=step_idx, phase="eval",
                    )

        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1
        avg_loss = total_loss / num_batches
        accuracy = exact_seq_correct / total_samples if total_samples > 0 else 0.0
        tok_accuracy = token_correct / total_tokens if total_tokens > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "accuracy": accuracy,
            "token_accuracy": tok_accuracy,
        }

    def _sync_time(self) -> float:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        return time.perf_counter()

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Optional[Iterable[Batch]] = None,
        *,
        epochs: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        epochs = epochs or self.config.epochs
        history: List[Dict[str, float]] = []
        
        if self.verbose:
            self.progress = TrainingProgress(epochs, sweep_info=self.sweep_info)
            self.progress.start()

        for listener in self.listeners:
            listener.on_train_start()

        try:
            for epoch in range(1, epochs + 1):
                epoch_start = self._sync_time()
                
                for listener in self.listeners:
                    listener.on_epoch_start(epoch)

                train_loss = self.train_epoch(train_loader, epoch)
                train_end = self._sync_time()
                eval_seconds = 0.0
                metrics = {"epoch": epoch, "train/loss": train_loss}

                if val_loader:
                    val_metrics = self.evaluate(val_loader)
                    eval_seconds = self._sync_time() - train_end
                    prefixed_val_metrics = {
                        f"val/{k}" if not k.startswith("val") else k: v
                        for k, v in val_metrics.items()
                    }
                    metrics.update(prefixed_val_metrics)
                    metrics["loss"] = train_loss
                    metrics.update(val_metrics)
                else:
                    metrics["loss"] = train_loss

                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        val_loss = metrics.get("val/loss", metrics.get("loss"))
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]
                metrics["lr"] = current_lr
                weight_decay = self.optimizer.param_groups[0].get("weight_decay", 0)
                metrics["weight_decay"] = weight_decay

                epoch_time = self._sync_time() - epoch_start
                metrics["time/epoch_seconds"] = epoch_time
                metrics["time/train_seconds"] = train_end - epoch_start
                metrics["time/eval_seconds"] = eval_seconds
                
                if self.progress and self.verbose:
                    self.progress.end_epoch(epoch, epoch_time, metrics)

                self._log_to_wandb(metrics)

                self._remember_best_validation(epoch, metrics)

                for listener in self.listeners:
                    listener.on_epoch_end(epoch, metrics)

                history.append(metrics)

                if self.config.early_stop_at_accuracy is not None and metrics.get("accuracy", 0) >= self.config.early_stop_at_accuracy:
                    if self.verbose:
                        msg = f"  Early stopping: accuracy {metrics['accuracy']:.2%} >= {self.config.early_stop_at_accuracy:.0%} threshold."
                        if RICH_AVAILABLE and _console:
                            _console.print(f"  [green]Early stopping:[/green] accuracy {metrics['accuracy']:.2%} >= {self.config.early_stop_at_accuracy:.0%} threshold.")
                        else:
                            print(msg)
                    break

            self.restore_best_validation()

            for listener in self.listeners:
                listener.on_train_end(history)
            
            if self.progress and self.verbose:
                self.progress.finish(history)

        finally:
            self._finish_wandb()

        return history

    def _remember_best_validation(self, epoch: int, metrics: Dict[str, float]) -> None:
        acc = metrics.get("val/accuracy", metrics.get("val_accuracy"))
        if acc is None:
            return
        acc = float(acc)
        if self._best_val_state is not None and acc <= self._best_val_accuracy:
            return
        self._best_val_accuracy = acc
        self.best_val_epoch = epoch
        self._best_val_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def restore_best_validation(self) -> bool:
        if self._best_val_state is None:
            return False
        self.model.load_state_dict(self._best_val_state)
        return True


import click

from sahi import __version__ as sahi_version
from sahi.predict import predict, predict_fiftyone
from sahi.scripts.coco2fiftyone import main as coco2fiftyone
from sahi.scripts.coco2yolo import main as coco2yolo
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate
from sahi.scripts.slice_coco import slicer
from sahi.utils.import_utils import print_environment_info


@click.group()
def cli():
    pass

coco_app = {
    "evaluate": evaluate,
    "analyse": analyse,
    "fiftyone": coco2fiftyone,
    "slice": slicer,
    "yolo": coco2yolo,
    "yolov5": coco2yolo,
}

sahi_app = {
    "predict": predict,
    "predict-fiftyone": predict_fiftyone,
    "coco": coco_app,
    "version": sahi_version,
    "env": print_environment_info,
}

def _make_callback(obj):
    """Return a callable suitable for click.Command:
    - if obj is callable, call it with whatever click passes;
    - otherwise print the object (e.g. version string).
    """
    if callable(obj):
        def _cb(*args, **kwargs):
            return obj(*args, **kwargs)
    else:
        def _cb(*args, **kwargs):
            click.echo(str(obj))
    return _cb

import inspect


def _click_params_from_signature(func):
    """Create a list of click.Parameter (Argument/Option) objects from a Python
    callable's signature. This provides a lightweight automatic mapping so
    CLI options are available without manually writing decorators.

    Rules (simple, pragmatic):
    - positional parameters without default -> click.Argument (required)
    - parameters with a default -> click.Option named --param-name
    - bool defaults -> is_flag option
    - list/tuple defaults -> multiple option
    - skip *args/**kwargs and (self, cls)
    - use annotation or default value to infer type when possible
    """
    params = []
    sig = inspect.signature(func)
    for name, p in sig.parameters.items():
        # skip common unrepresentable params
        if name in ("self", "cls"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # skip *args/**kwargs
            continue

        if p.default is inspect._empty:
            # required positional argument
            params.append(click.Argument([name]))
        else:
            opt_name = f"--{name}"
            # boolean flags
            if isinstance(p.default, bool):
                params.append(click.Option([opt_name], is_flag=True, default=p.default, help=f"(auto) default={p.default}"))
            # lists/tuples -> multiple
            elif isinstance(p.default, (list, tuple)):
                params.append(click.Option([opt_name], multiple=True, default=tuple(p.default), help="(auto) multiple"))
            else:
                # infer type from annotation or default value
                param_type = None
                if p.annotation is not inspect._empty and p.annotation in (int, float, str, bool):
                    param_type = p.annotation
                elif p.default is not None:
                    param_type = type(p.default)
                else:
                    param_type = str
                params.append(click.Option([opt_name], default=p.default, type=param_type, help=f"(auto) default={p.default}"))
    return params


def _make_click_command(name, func):
    """Build a click.Command for `func`, auto-generating params from signature if callable."""
    params = _click_params_from_signature(func) if callable(func) else []
    return click.Command(name, params=params, callback=_make_callback(func))


def app() -> None:
    """Cli app."""
    #fire.Fire(sahi_app)
    # for loop to add commands to cli
    for command_name, command_func in sahi_app.items():
        if isinstance(command_func, dict):
            # add subcommands
            sub_cli = click.Group(command_name)
            for sub_command_name, sub_command_func in command_func.items():
                sub_cli.add_command(_make_click_command(sub_command_name, sub_command_func))
            cli.add_command(sub_cli)
        else:
            cli.add_command(_make_click_command(command_name, command_func))

    cli()

if __name__ == "__main__":
    cli()
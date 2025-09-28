import inspect

import click


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

        opt_name = f"--{name}"
        
        if p.default is inspect._empty:
            # required option (no default value)
            param_type = None
            if p.annotation is not inspect._empty and p.annotation in (int, float, str, bool):
                param_type = p.annotation
            else:
                param_type = str
            params.append(
                click.Option([opt_name], required=True, type=param_type, help="(auto)")
            )
        else:
            # boolean flags
            if isinstance(p.default, bool):
                params.append(
                    click.Option([opt_name], is_flag=True, default=p.default, help=f"(auto) default={p.default}")
                )
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
                params.append(
                    click.Option([opt_name], default=p.default, type=param_type, help=f"(auto) default={p.default}")
                )
    return params


def make_click_command(name, func):
    """Build a click.Command for `func`, auto-generating params from signature if callable."""
    params = _click_params_from_signature(func) if callable(func) else []
    # use function docstring as help when available, but only the first line (summary)
    help_text = None
    if callable(func):
        docstring = func.__doc__ if getattr(func, "__doc__", None) else None
        if docstring:
            # Extract only the first line of the docstring for cleaner CLI help
            help_text = docstring.strip().split("\n")[0]
    return click.Command(name, params=params, callback=_make_callback(func), help=help_text)

import click

from sahi.utils.cli_helper import make_click_command


# Truly lazy import wrapper functions - only import when called
def predict_command(*args, **kwargs):
    """Perform prediction - imports heavy dependencies only when called."""
    from sahi.predict import predict

    return predict(*args, **kwargs)


def predict_fiftyone_command(*args, **kwargs):
    """Perform FiftyOne prediction - imports heavy dependencies only when called."""
    from sahi.predict import predict_fiftyone

    return predict_fiftyone(*args, **kwargs)


def coco_evaluate_command(*args, **kwargs):
    """COCO evaluation - imports dependencies only when called."""
    from sahi.scripts.coco_evaluation import evaluate

    return evaluate(*args, **kwargs)


def coco_analyse_command(*args, **kwargs):
    """COCO analysis - imports dependencies only when called."""
    from sahi.scripts.coco_error_analysis import analyse

    return analyse(*args, **kwargs)


def coco_fiftyone_command(*args, **kwargs):
    """COCO to FiftyOne conversion - imports dependencies only when called."""
    from sahi.scripts.coco2fiftyone import main

    return main(*args, **kwargs)


def coco_slice_command(*args, **kwargs):
    """COCO slicing - imports dependencies only when called."""
    from sahi.scripts.slice_coco import slicer

    return slicer(*args, **kwargs)


def coco_yolo_command(*args, **kwargs):
    """COCO to YOLO conversion - imports dependencies only when called."""
    from sahi.scripts.coco2yolo import main

    return main(*args, **kwargs)


@click.group(help="SAHI command-line utilities: slicing-aided high-resolution inference and COCO tools")
def cli():
    """Top-level click group for SAHI CLI."""
    pass


# Create wrapper functions with proper help text for commands that need it
def version_command():
    """Show SAHI version."""
    from sahi import __version__ as sahi_version

    click.echo(sahi_version)


def env_command():
    """Show environment information - imports dependencies only when called."""
    from sahi.utils.package_utils import print_environment_info

    print_environment_info()


def app() -> None:
    """Cli app."""

    # Define coco subcommands with truly lazy imports
    coco_functions = {
        "evaluate": coco_evaluate_command,
        "analyse": coco_analyse_command,
        "fiftyone": coco_fiftyone_command,
        "slice": coco_slice_command,
        "yolo": coco_yolo_command,
        "yolov5": coco_yolo_command,  # yolov5 is an alias for yolo
    }

    sahi_app = {
        "predict": predict_command,
        "predict-fiftyone": predict_fiftyone_command,
        "coco": coco_functions,
        "version": version_command,
        "env": env_command,
    }

    for command_name, command_func in sahi_app.items():
        if isinstance(command_func, dict):
            # add subcommands (create a named Group with help text)
            sub_cli = click.Group(command_name, help=f"{command_name} related commands")
            for sub_command_name, sub_command_func in command_func.items():
                sub_cli.add_command(make_click_command(sub_command_name, sub_command_func))
            cli.add_command(sub_cli)
        else:
            cli.add_command(make_click_command(command_name, command_func))

    cli()


if __name__ == "__main__":
    # build the application (register commands) and run
    app()

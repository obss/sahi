import click

from sahi import __version__ as sahi_version
from sahi.predict import predict, predict_fiftyone
from sahi.scripts.coco2fiftyone import main as coco2fiftyone
from sahi.scripts.coco2yolo import main as coco2yolo
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate
from sahi.scripts.slice_coco import slicer
from sahi.utils.cli_helper import make_click_command
from sahi.utils.package_utils import print_environment_info


@click.group(help="SAHI command-line utilities: slicing-aided high-resolution inference and COCO tools")
def cli():
    """Top-level click group for SAHI CLI."""
    pass


coco_app = {
    "evaluate": evaluate,
    "analyse": analyse,
    "fiftyone": coco2fiftyone,
    "slice": slicer,
    "yolo": coco2yolo,
    "yolov5": coco2yolo,
}


# Create wrapper functions with proper help text for commands that need it
def version_command():
    """Show SAHI version."""
    click.echo(sahi_version)


def env_command():
    """Show environment information."""
    print_environment_info()


sahi_app = {
    "predict": predict,
    "predict-fiftyone": predict_fiftyone,
    "coco": coco_app,
    "version": version_command,
    "env": env_command,
}


def app() -> None:
    """Cli app."""

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

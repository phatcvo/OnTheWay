import os
import sys

from gymnasium.envs.registration import register

__version__ = "1.1.2"

try:
    from farama_notifications import notifications

    if "otw_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["otw_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass


# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_otw_envs():
    """Import the envs module so that envs register themselves."""


    # street_env.py
    register(
        id="street-v1",
        entry_point="otw_env.envs.street_env:StreetEnv",
    )

_register_otw_envs()
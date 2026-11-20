''' Passive scalar (contrast) transport on a fluid mesh: `ScalarTransport` (the PDE, velocity
supplied by the caller), `StoredVelocity` (a finished run's velocity, interpolated in time),
and the offline driver `python -m nsx.transport.offline config.yaml`. '''
from .scalar import ScalarTransport  # noqa: F401
from .velocity import StoredVelocity, CTVelocity  # noqa: F401

# Autoware Wrapper

This wrapper exposes the Autoware driving stack through the PISA AV service.
The wrapper and runner must use compatible versions of `pisa-api` supporting
the Ping/Init identity contract (`pisa-api>=0.4.1` for this package).

## Ping and initialization identity

`Pong.name` is the stable wrapper artifact identity, `autoware-wrapper`, and
`Pong.version` is the installed package/build version (currently `0.3.1`). The
version is read from the `autoware-wrapper` distribution metadata, with the
repository's `[project].version` used only when running from a source checkout.

After initialization, `InitResponse.name` is `autoware`, the canonical name of
the driving stack that was successfully initialized. This differs deliberately
from the wrapper artifact name.

`InitResponse.metadata.effective_config` contains the normalized
wrapper-specific Autoware settings that actually took effect. For example:

```json
{
  "effective_config": {
    "launch": {
      "package": "autoware_launch",
      "file": "pisa.launch.xml",
      "headless": true,
      "extra_args": [],
      "log": false,
      "log_path": "autoware_launch.log"
    },
    "vehicle": {
      "model": "sample_vehicle",
      "sensor_model": "sample_sensor_kit"
    },
    "runtime": {
      "publish_agent_objects": true,
      "timeout_sec": 30.0
    }
  }
}
```

The execution manifest records this metadata. Never put secrets, tokens,
credentials, environment variables, or unfiltered raw config in it. Shared
execution data such as `dt`, map identity, scenario identity, and output paths
belongs to the runner/manifest and is not duplicated in wrapper metadata.

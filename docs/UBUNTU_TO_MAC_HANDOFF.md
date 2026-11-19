# Ubuntu to Mac Handoff

The bundle is built at:

`handoff/contact_memory_diagnostics_seed14011/`

Verify it on Ubuntu before transfer:

```bash
python scripts/check_artifact_integrity.py handoff/contact_memory_diagnostics_seed14011/manifest.json
sha256sum handoff/contact_memory_diagnostics_seed14011.tar.zst
```

Inbound SSH on Ubuntu is inactive. Tailscale is authenticated and the Mac node is currently directly reachable, so Taildrop is the preferred non-public path:

```bash
tailscale file cp handoff/contact_memory_diagnostics_seed14011.tar.zst 2:
```

On macOS, accept the file using the Tailscale UI (or `tailscale file get ~/Downloads` where supported), then run:

```bash
cd ~/Downloads
sha256sum -c contact_memory_diagnostics_seed14011.tar.zst.sha256
tar --use-compress-program=unzstd -xf contact_memory_diagnostics_seed14011.tar.zst
cd contact_memory_diagnostics_seed14011
python scripts/check_artifact_integrity.py manifest.json
```

No transfer was automatically initiated. This avoids placing private model bytes on a public service and lets the user choose the Mac destination. If Taildrop is unavailable, enable Remote Login on the Mac and push outbound over its Tailscale address with `rsync -aP --partial`.

Real fixed-action memory diagnostics on Mac require a compatible PyTorch environment:

```bash
python scripts/run_real_memory_interventions.py --checkpoint checkpoints/m2_v2/model_12000.pt --trajectory trajectories/alternating.pt --trajectory-sha256 d5f0d32a172605e2816ce740069dd4ce0e5f4c93416152b9e41954907bc8b651 --condition alternating --environment 8 --start 0 --output validation/mac_real_memory_interventions.csv
```

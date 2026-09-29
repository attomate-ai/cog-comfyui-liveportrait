# Smoke-test inputs

Used only by `.github/workflows/smoke.yml`. Not part of the cog image.

- `portrait.jpg`: `assets/examples/source/s6.jpg` from
  [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) at commit
  `9b294b3d0536135442ea73cb01e6cb3ca7029dd3`, unmodified. MIT License,
  Copyright (c) 2024 Kuaishou Visual Generation and Interaction Center.
- `driving.mp4`: the first 5 s of `assets/examples/driving/d20.mp4` from the same
  repo and commit (MIT, same copyright), converted with VidStudio's driving-clip
  filter chain: `fps=30,crop='min(iw,ih)':'min(iw,ih)',scale=512:512`, H.264
  CRF 18, yuv420p, no audio. 150 frames at 30 fps.

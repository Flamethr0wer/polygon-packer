# Flamethrower's polygon packer
This program can quickly solve the 2D bin packing problem for any number of any polygons inside any other polygon! It was the tool used to find all the optimal packings under the name "Ignacio Vallejo" on [Erich's Packing Center](https://erich-friedman.github.io/packing/).
<img width="640" height="480" alt="30 triangles in a hexagon" src="https://github.com/user-attachments/assets/48591a93-3ed9-4031-9c42-8b6eb579d91e" />

## How to use
To use it, download the python file, navigate to its location and run it like this:

`python3 polygon_packer.py [n] [nsi] [nsc]`
- Replace `[n]` with the number of inner polygons you want to solve for
- Replace `[nsi]` with the number of sides of the inner polygons (e.g. 4 for a square)
- Replace `[nsc]` with the number of sides of the container polygon

Optional parameters:
- `--attempts`: the total number of attempts to run. Increase to explore more possible packings. Defaults to 1000.
- `--tolerance`: the tolerance for the penalty function. More penalty reduces the margin of overlap but limits exporation. Defaults to an empirical sweetspot of 0.00000001.
- `--finalstep`: the container size is decreased by a smaller factor each time, to save compute at the beginning and achieve greater precision near the end. This sets the step size of the shrinkage which would correspond to the theoretical minimum container size (which, for most packings, will not actually be reached, so keep that in mind when setting this parameter). Defaults to 0.0001.

# Polygon Packer - JavaScript Version

This is a JavaScript rewrite of the polygon packing algorithm. The goal is to port the Python implementation to JavaScript for faster execution and better compatibility.

## Getting Started

### Installation

```bash
npm install
```

### Usage

```bash
node polygon_packer.js [n] [nsi] [nsc]
```

- Replace `[n]` with the number of inner polygons you want to solve for
- Replace `[nsi]` with the number of sides of the inner polygons (e.g., 4 for a square)
- Replace `[nsc]` with the number of sides of the container polygon

### Optional Parameters

- `--attempts`: The total number of attempts to run. Increase to explore more possible packings. Defaults to 1000.
- `--tolerance`: The tolerance for the penalty function. Defaults to 1e-8.
- `--finalstep`: The container size decrease step size. Defaults to 0.0001.

### Example

```bash
node polygon_packer.js 30 3 6
```

This will attempt to pack 30 triangles into a hexagon.

## Implementation Notes

- The JavaScript version uses the Separating Axis Theorem (SAT) for efficient polygon collision detection
- Optimization uses a simulated annealing-like approach with decreasing container scale
- The implementation aims to replicate the Python version's functionality while leveraging JavaScript's event-driven paradigm

## Development

Tests can be run with:

```bash
npm test
```

## Status

This is an active rewrite. Core functionality is being developed and tested incrementally.

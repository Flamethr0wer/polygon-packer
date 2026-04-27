import numeric from 'numeric';
import { createCanvas } from 'canvas';
import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
const argsObj = {
  innerPolygons: parseInt(args[0]) || 10,
  innerSides: parseInt(args[1]) || 3,
  containerSides: parseInt(args[2]) || 6,
  attempts: 1000,
  tolerance: 1e-8,
  finalStep: 0.0001
};

// Parse optional arguments
for (let i = 3; i < args.length; i++) {
  if (args[i] === '--attempts' && i + 1 < args.length) {
    argsObj.attempts = parseInt(args[i + 1]);
    i++;
  } else if (args[i] === '--tolerance' && i + 1 < args.length) {
    argsObj.tolerance = parseFloat(args[i + 1]);
    i++;
  } else if (args[i] === '--finalstep' && i + 1 < args.length) {
    argsObj.finalStep = parseFloat(args[i + 1]);
    i++;
  }
}

const {
  innerPolygons: N,
  innerSides: nsi,
  containerSides: nsc,
  attempts,
  tolerance: penaltyTolerance,
  finalStep: finalStepSize
} = argsObj;

console.log(`Polygon Packer - JavaScript Version`);
console.log(`Packing ${N} ${nsi}-sided polygons into a ${nsc}-sided container`);
console.log(`Attempts: ${attempts}, Tolerance: ${penaltyTolerance}, Final Step: ${finalStepSize}`);

/**
 * Generate unit polygon vertices at angle intervals
 */
function generatePolygonVertices(sides) {
  const angles = [];
  for (let i = 0; i < sides; i++) {
    angles.push((2 * Math.PI * i) / sides);
  }
  
  return angles.map(angle => [
    Math.cos(angle),
    Math.sin(angle)
  ]);
}

/**
 * Generate unit polygon normal vectors (perpendicular to edges)
 */
function generatePolygonVectors(sides) {
  const angles = [];
  for (let i = 0; i < sides; i++) {
    angles.push((2 * Math.PI * i) / sides + Math.PI / sides);
  }
  
  return angles.map(angle => [
    Math.cos(angle),
    Math.sin(angle)
  ]);
}

/**
 * Transform a polygon by translation and rotation
 */
function transformPolygon(x, y, angle, vertices) {
  const cos_a = Math.cos(angle);
  const sin_a = Math.sin(angle);
  
  return vertices.map(([vx, vy]) => [
    x + (vx * cos_a - vy * sin_a),
    y + (vx * sin_a + vy * cos_a)
  ]);
}

/**
 * Rotate normal vectors by an angle
 */
function rotateVectors(angle, vectors) {
  const cos_a = Math.cos(angle);
  const sin_a = Math.sin(angle);
  
  return vectors.map(([vecx, vecy]) => [
    vecx * cos_a - vecy * sin_a,
    vecx * sin_a + vecy * cos_a
  ]);
}

/**
 * Calculate the signed distance from a point to a line (defined by a point and direction)
 */
function signedDistance(px, py, ax, ay, dirx, diry) {
  return (px - ax) * (-diry) + (py - ay) * dirx;
}

/**
 * Check if two convex polygons overlap using Separating Axis Theorem
 */
function polygonsOverlap(poly1, poly2, axes) {
  for (const [ax, ay] of axes) {
    let min1 = Infinity, max1 = -Infinity;
    let min2 = Infinity, max2 = -Infinity;
    
    for (const [vx, vy] of poly1) {
      const proj = vx * ax + vy * ay;
      min1 = Math.min(min1, proj);
      max1 = Math.max(max1, proj);
    }
    
    for (const [vx, vy] of poly2) {
      const proj = vx * ax + vy * ay;
      min2 = Math.min(min2, proj);
      max2 = Math.max(max2, proj);
    }
    
    // Check for gap on this axis
    if (max1 < min2 - 1e-10 || max2 < min1 - 1e-10) {
      return false; // No overlap
    }
  }
  
  return true; // Overlap detected on all axes
}

/**
 * Check if a polygon is inside another polygon (container)
 */
function polygonInside(polygon, container, containerVectors) {
  for (const [px, py] of polygon) {
    for (let i = 0; i < container.length; i++) {
      const [ax, ay] = container[i];
      const [dirx, diry] = containerVectors[i];
      
      const dist = signedDistance(px, py, ax, ay, dirx, diry);
      if (dist < -1e-10) {
        return false; // Point is outside
      }
    }
  }
  return true; // All points inside
}

/**
 * Calculate overlap penalty between polygons
 */
function calculateOverlapPenalty(polygons, transformedPolygons, unitVectors, axes) {
  let penalty = 0;
  
  for (let i = 0; i < transformedPolygons.length; i++) {
    for (let j = i + 1; j < transformedPolygons.length; j++) {
      if (polygonsOverlap(transformedPolygons[i], transformedPolygons[j], axes)) {
        penalty += 1.0 / penaltyTolerance;
      }
    }
  }
  
  return penalty;
}

/**
 * Main packing algorithm
 */
function packPolygons() {
  const unitInnerVertices = generatePolygonVertices(nsi);
  const unitInnerVectors = generatePolygonVectors(nsi);
  const unitContainerVertices = generatePolygonVertices(nsc);
  const unitContainerVectors = generatePolygonVectors(nsc);
  const unitContainerApothem = Math.cos(Math.PI / nsc);
  
  // Generate all normal vectors for SAT
  const allAxes = [...unitInnerVectors, ...unitContainerVectors];
  
  // Initial random configuration
  let bestConfiguration = null;
  let bestContainerScale = Infinity;
  
  for (let attempt = 0; attempt < attempts; attempt++) {
    if ((attempt + 1) % 100 === 0) {
      console.log(`Attempt ${attempt + 1}/${attempts}...`);
    }
    
    // Generate random positions and rotations
    const configuration = [];
    for (let i = 0; i < N; i++) {
      configuration.push({
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 2,
        angle: Math.random() * 2 * Math.PI
      });
    }
    
    // Try to fit all polygons with decreasing container scale
    let containerScale = 2.0;
    let validFit = true;
    
    while (containerScale > finalStepSize && validFit) {
      const scaledContainerVertices = unitContainerVertices.map(([vx, vy]) => [
        vx * containerScale,
        vy * containerScale
      ]);
      const scaledContainerVectors = unitContainerVectors.map(([vx, vy]) => [
        vx * containerScale,
        vy * containerScale
      ]);
      
      validFit = true;
      for (let i = 0; i < N; i++) {
        const transformedInner = transformPolygon(
          configuration[i].x,
          configuration[i].y,
          configuration[i].angle,
          unitInnerVertices
        );
        
        if (!polygonInside(transformedInner, scaledContainerVertices, scaledContainerVectors)) {
          validFit = false;
          break;
        }
      }
      
      if (validFit) {
        containerScale *= 0.95; // Decrease by 5%
      }
    }
    
    if (validFit && containerScale < bestContainerScale) {
      bestContainerScale = containerScale;
      bestConfiguration = {
        configuration,
        scale: containerScale
      };
      console.log(`Found better packing at scale: ${containerScale.toFixed(6)}`);
    }
  }
  
  if (bestConfiguration) {
    console.log(`\nBest packing found!`);
    console.log(`Container scale: ${bestConfiguration.scale.toFixed(6)}`);
    console.log(`Container apothem: ${(bestConfiguration.scale * unitContainerApothem).toFixed(6)}`);
  } else {
    console.log(`No valid packing found after ${attempts} attempts.`);
  }
  
  return bestConfiguration;
}

// Run the packing algorithm
const result = packPolygons();

if (result) {
  console.log(`\nConfiguration:`);
  console.log(JSON.stringify(result, null, 2));
}

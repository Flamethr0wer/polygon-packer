extern crate argmin;
extern crate clap;
extern crate ndarray;
extern crate plotters;
extern crate rayon;

mod rust_specific;
use rust_specific::AssocPI;
use rust_specific::FloatType;
// this is π in either f32 or f64
const PI: FloatType = <FloatType as AssocPI>::PI;

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;
use clap::Parser;
use ndarray::Array2;
use plotters::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use voxell_rng::rng::pcg_advanced::pcg_64::PcgInnerState64;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of inner polygons
    inner_polygons: usize,
    /// Number of sides of the inner polygons
    inner_sides: usize,
    /// Number of sides of the container polygon
    container_sides: usize,
    /// Number of attempts to run
    #[arg(long, default_value_t = 1000)]
    attempts: usize,
    /// Overlap penalty tolerance. Probably best left at default
    #[arg(long, default_value_t = 1e-8)]
    tolerance: FloatType,
    /// How small the last theoretical step in container size decrease will be (it gets smaller over time)
    #[arg(long, default_value_t = 0.0001)]
    finalstep: FloatType,
}

fn main() {
    let args = Args::parse();

    let N = args.inner_polygons;
    let nsi = args.inner_sides;
    let nsc = args.container_sides;
    let attempts = args.attempts;
    let penalty_tolerance = args.tolerance;
    let final_step_size = args.finalstep;

    // Translated from Python:
    // unit_polygon_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
    let unit_polygon_angles_vec: Vec<FloatType> = (0..nsi)
        .map(|i| 2.0 * PI * (i as FloatType) / (nsi as FloatType))
        .collect();

    // unit_polygon_vertices = np.column_stack((np.cos(unit_polygon_angles), np.sin(unit_polygon_angles)))
    let mut unit_polygon_vertices_buf: Vec<FloatType> = Vec::with_capacity(nsi * 2);
    for &a in &unit_polygon_angles_vec {
        unit_polygon_vertices_buf.push(a.cos());
        unit_polygon_vertices_buf.push(a.sin());
    }
    let unit_polygon_vertices: Array2<FloatType> =
        Array2::from_shape_vec((nsi, 2), unit_polygon_vertices_buf).unwrap();

    // unit_polygon_vectors = np.column_stack((np.cos(unit_polygon_angles + np.pi / nsi), np.sin(unit_polygon_angles + np.pi / nsi)))
    let mut unit_polygon_vectors_buf: Vec<FloatType> = Vec::with_capacity(nsi * 2);
    let offset_nsi = PI / (nsi as FloatType);
    for &a in &unit_polygon_angles_vec {
        let aa = a + offset_nsi;
        unit_polygon_vectors_buf.push(aa.cos());
        unit_polygon_vectors_buf.push(aa.sin());
    }
    let unit_polygon_vectors: Array2<FloatType> =
        Array2::from_shape_vec((nsi, 2), unit_polygon_vectors_buf).unwrap();

    // unit_container_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
    let unit_container_angles: Vec<FloatType> = (0..nsc)
        .map(|i| 2.0 * PI * (i as FloatType) / (nsc as FloatType))
        .collect();

    // unit_container_vertices = np.column_stack((np.cos(unit_container_angles), np.sin(unit_container_angles)))
    let mut unit_container_vertices_buf: Vec<FloatType> = Vec::with_capacity(nsc * 2);
    for &a in &unit_container_angles {
        unit_container_vertices_buf.push(a.cos());
        unit_container_vertices_buf.push(a.sin());
    }
    let unit_container_vertices: Array2<FloatType> =
        Array2::from_shape_vec((nsc, 2), unit_container_vertices_buf).unwrap();

    // unit_container_vectors = np.column_stack((np.cos(unit_container_angles + np.pi / nsc), np.sin(unit_container_angles + np.pi / nsc)))
    let mut unit_container_vectors_buf: Vec<FloatType> = Vec::with_capacity(nsc * 2);
    let offset_nsc = PI / (nsc as FloatType);
    for &a in &unit_container_angles {
        let aa = a + offset_nsc;
        unit_container_vectors_buf.push(aa.cos());
        unit_container_vectors_buf.push(aa.sin());
    }
    let unit_container_vectors: Array2<FloatType> =
        Array2::from_shape_vec((nsc, 2), unit_container_vectors_buf).unwrap();

    // unit_container_apothem = np.cos(np.pi / nsc)
    let unit_container_apothem: FloatType = (PI / (nsc as FloatType)).cos();

    let mut best_S: FloatType = FloatType::INFINITY;
    let mut best_values: Option<Vec<FloatType>> = None;

    // results = Parallel(n_jobs=-1, prefer="processes")(delayed(repetition)(i) for i in range(attempts))
    let results: Vec<(FloatType, _)> = (0..attempts)
        .into_par_iter()
        .map(|i| {
            repetition(
                i,
                N,
                nsi,
                nsc,
                penalty_tolerance,
                final_step_size,
                &unit_polygon_vertices,
                &unit_polygon_vectors,
                &unit_container_vectors,
                unit_container_apothem,
            )
        })
        .collect();

    for (s, values) in results {
        if s < best_S {
            best_S = s;
            best_values = Some(values);
        }
    }

    println!(
        "Final side length: {}",
        // best_S * np.sin(np.pi / nsc) / np.sin(np.pi / nsi),
        best_S * (PI / (nsc as FloatType)).sin() / (PI / (nsi as FloatType)).sin(),
    );

    // ============== NOTE(@paladynee): ===============
    // anything found after this line is translated from Python into Rust
    // via an AI agent. the numerical results (excluding nondeterminism from
    // rngs) have been checked against python over multiple passes.
    //
    // i am not well versed in mathematics so if anybody could fact check this
    // implementation and remove this note, I'd be very glad.  

    if let Some(vals) = best_values {
        if vals.len() == N * 3 {
            let positions = Array2::from_shape_vec((N, 3), vals.clone()).unwrap();
            println!("Final positions (first 5 rows):");
            for i in 0..std::cmp::min(5, N) {
                println!("  {}: {:?}", i, positions.row(i));
            }

            // Plot result to PNG using Plotters (match Python output)
            let file_name = format!("{}_{}_in_{}.png", N, nsi, nsc);
            // collect polygon points and container points (as f64)
            let mut all_points: Vec<(f64, f64)> = Vec::new();
            let mut polys: Vec<Vec<(f64, f64)>> = Vec::with_capacity(N);
            for i in 0..N {
                let x = positions[[i, 0]];
                let y = positions[[i, 1]];
                let a = positions[[i, 2]];
                let poly = transform_polygon(x, y, a, &unit_polygon_vertices);
                let mut pts: Vec<(f64, f64)> = Vec::with_capacity(poly.len() / 2 + 1);
                for k in 0..(poly.len() / 2) {
                    let px = poly[k * 2] as f64;
                    let py = poly[k * 2 + 1] as f64;
                    pts.push((px, py));
                    all_points.push((px, py));
                }
                if let Some(first) = pts.get(0).cloned() {
                    pts.push(first);
                    all_points.push(first);
                }
                polys.push(pts);
            }

            let mut container_pts: Vec<(f64, f64)> =
                Vec::with_capacity(unit_container_vertices.shape()[0] + 1);
            for i in 0..unit_container_vertices.shape()[0] {
                let cx = unit_container_vertices[[i, 0]] * best_S;
                let cy = unit_container_vertices[[i, 1]] * best_S;
                container_pts.push((cx as f64, cy as f64));
                all_points.push((cx as f64, cy as f64));
            }
            if let Some(first) = container_pts.get(0).cloned() {
                container_pts.push(first);
                all_points.push(first);
            }

            // determine bounds and padding
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (
                std::f64::INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::INFINITY,
                std::f64::NEG_INFINITY,
            );
            for (x, y) in &all_points {
                if *x < min_x {
                    min_x = *x;
                }
                if *x > max_x {
                    max_x = *x;
                }
                if *y < min_y {
                    min_y = *y;
                }
                if *y > max_y {
                    max_y = *y;
                }
            }
            // choose grid steps matching the requested ticks
            let x_step = 1.0_f64;
            let y_step = 0.5_f64;
            let x_min_tick = (min_x / x_step).floor() * x_step;
            let x_max_tick = (max_x / x_step).ceil() * x_step;
            let y_min_tick = (min_y / y_step).floor() * y_step;
            let y_max_tick = (max_y / y_step).ceil() * y_step;

            // Equalize spans so 1 unit in x equals 1 unit in y on the image
            let dx = x_max_tick - x_min_tick;
            let dy = y_max_tick - y_min_tick;
            let span = dx.max(dy);
            let x_center = (x_min_tick + x_max_tick) / 2.0;
            let y_center = (y_min_tick + y_max_tick) / 2.0;
            let x_min_eq = x_center - span / 2.0;
            let x_max_eq = x_center + span / 2.0;
            let y_min_eq = y_center - span / 2.0;
            let y_max_eq = y_center + span / 2.0;

            // Render PNG on a square canvas so aspect ratio is preserved
            let backend = BitMapBackend::new(&file_name, (1024, 1024));
            let root = backend.into_drawing_area();
            root.fill(&WHITE).unwrap();
            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .caption(
                    format!(
                        "Side length: {}",
                        best_S * (PI / (nsc as FloatType)).sin() / (PI / (nsi as FloatType)).sin()
                    ),
                    ("sans-serif", 20).into_font(),
                )
                .build_cartesian_2d(x_min_eq..x_max_eq, y_min_eq..y_max_eq)
                .unwrap();

            // configure mesh to show ticks at our step sizes
            let x_labels = ((span / x_step).round() as usize).saturating_add(1).max(2);
            let y_labels = ((span / y_step).round() as usize).saturating_add(1).max(2);
            chart
                .configure_mesh()
                .x_labels(x_labels)
                .y_labels(y_labels)
                .x_label_formatter(&|x| format!("{:.0}", x))
                .y_label_formatter(&|y| format!("{:.1}", y))
                .draw()
                .ok();

            // draw container outline
            chart
                .draw_series(std::iter::once(PathElement::new(
                    container_pts.clone(),
                    &BLACK,
                )))
                .ok();

            // draw filled polygons and outlines
            for poly in polys {
                let fill_style = Into::<ShapeStyle>::into(&RGBColor(204, 204, 204)).filled();
                chart
                    .draw_series(std::iter::once(Polygon::new(poly.clone(), fill_style)))
                    .ok();
                chart
                    .draw_series(std::iter::once(PathElement::new(poly, &BLACK)))
                    .ok();
            }

            root.present().ok();
            println!("Saved plot to {}", file_name);
        }
    } else {
        println!("No valid packing found in attempts");
    }
}

// def repetition(seed):
fn transform_polygon(
    x: FloatType,
    y: FloatType,
    a: FloatType,
    vertices: &Array2<FloatType>,
) -> Vec<FloatType> {
    let n_vertices = vertices.shape()[0];
    let mut transformed = vec![0.0 as FloatType; n_vertices * 2];
    let ca = a.cos();
    let sa = a.sin();
    for i in 0..n_vertices {
        let vx = vertices[[i, 0]];
        let vy = vertices[[i, 1]];
        transformed[i * 2] = x + (vx * ca - vy * sa);
        transformed[i * 2 + 1] = y + (vx * sa + vy * ca);
    }
    transformed
}



// BHProblem struct must be declared before its impl. It holds arena-backed
// scratch buffers to avoid transient allocations during cost evaluation.
struct BHProblem<'a, 'b> {
    N: usize,
    nsi: usize,
    nsc: usize,
    unit_polygon_vertices: &'a Array2<FloatType>,
    unit_polygon_vectors: &'a Array2<FloatType>,
    unit_container_vectors: &'a Array2<FloatType>,
    unit_container_apothem: FloatType,
    S: FloatType,
    arena: &'b Bump,
    polys_x: Rc<RefCell<BumpVec<'b, FloatType>>>,
    polys_y: Rc<RefCell<BumpVec<'b, FloatType>>>,
    vecs_x: Rc<RefCell<BumpVec<'b, FloatType>>>,
    vecs_y: Rc<RefCell<BumpVec<'b, FloatType>>>,
}

// BH cost evaluation implemented as a method on the problem struct.
// Use a bump arena and SoA scratch buffers (polys_x/polys_y, vecs_x/vecs_y)
// to reduce transient heap allocations and improve data locality.
impl<'a, 'b> BHProblem<'a, 'b> {
    fn bh_function(&self, values: &[FloatType]) -> FloatType {
        const HUGE: FloatType = 1e20 as FloatType;

        let n_vertices = self.unit_polygon_vertices.shape()[0];
        let nsi = self.nsi;
        let nsc = self.nsc;

        let limit = self.unit_container_apothem * self.S;

        let mut penalty: FloatType = 0.0;

        // Reusable, arena-allocated SoA buffers.
        let mut polys_x = self.polys_x.borrow_mut();
        let mut polys_y = self.polys_y.borrow_mut();
        let mut vecs_x = self.vecs_x.borrow_mut();
        let mut vecs_y = self.vecs_y.borrow_mut();

        polys_x.clear();
        polys_y.clear();
        vecs_x.clear();
        vecs_y.clear();

        // Reserve expected capacity to avoid repeated grows
        let total_points = (self.N as usize) * (n_vertices as usize);
        polys_x.reserve(total_points);
        polys_y.reserve(total_points);
        vecs_x.reserve(total_points);
        vecs_y.reserve(total_points);

        // Fill transformed polygon coordinates and rotated edge vectors.
        for i in 0..self.N {
            let posx = values[i * 3];
            let posy = values[i * 3 + 1];
            let rot = values[i * 3 + 2];
            let ca = rot.cos();
            let sa = rot.sin();

            for v in 0..n_vertices {
                let vx = self.unit_polygon_vertices[[v, 0]];
                let vy = self.unit_polygon_vertices[[v, 1]];
                let px = posx + (vx * ca - vy * sa);
                let py = posy + (vx * sa + vy * ca);
                polys_x.push(px);
                polys_y.push(py);

                // poking penalty against container
                for ci in 0..nsc {
                    let dvx = self.unit_container_vectors[[ci, 0]];
                    let dvy = self.unit_container_vectors[[ci, 1]];
                    let distance = px * dvx + py * dvy;
                    if distance > limit {
                        let diff = distance - limit;
                        penalty += diff * diff;
                    }
                }
            }

            for v in 0..n_vertices {
                let vx = self.unit_polygon_vectors[[v, 0]];
                let vy = self.unit_polygon_vectors[[v, 1]];
                vecs_x.push(vx * ca - vy * sa);
                vecs_y.push(vx * sa + vy * ca);
            }
        }

        // Pairwise separating axis test using SoA buffers
        for i in 0..self.N {
            for j in (i + 1)..self.N {
                let mut collision = true;
                let mut min_overlap: FloatType = HUGE;

                for vec_idx in 0..(nsi * 2) {
                    let (x_axis, y_axis) = if vec_idx < nsi {
                        (
                            vecs_x[i * nsi + vec_idx],
                            vecs_y[i * nsi + vec_idx],
                        )
                    } else {
                        (
                            vecs_x[j * nsi + (vec_idx - nsi)],
                            vecs_y[j * nsi + (vec_idx - nsi)],
                        )
                    };

                    let mut min_1 = HUGE;
                    let mut max_1 = -HUGE;
                    for vert in 0..nsi {
                        let dotp = polys_x[i * nsi + vert] * x_axis
                            + polys_y[i * nsi + vert] * y_axis;
                        if dotp < min_1 {
                            min_1 = dotp;
                        }
                        if dotp > max_1 {
                            max_1 = dotp;
                        }
                    }

                    let mut min_2 = HUGE;
                    let mut max_2 = -HUGE;
                    for vert in 0..nsi {
                        let dotp = polys_x[j * nsi + vert] * x_axis
                            + polys_y[j * nsi + vert] * y_axis;
                        if dotp < min_2 {
                            min_2 = dotp;
                        }
                        if dotp > max_2 {
                            max_2 = dotp;
                        }
                    }

                    let overlap = (if max_1 < max_2 { max_1 } else { max_2 })
                        - (if min_1 > min_2 { min_1 } else { min_2 });
                    if overlap <= (0.0 as FloatType) {
                        collision = false;
                        break;
                    }
                    if overlap < min_overlap {
                        min_overlap = overlap;
                    }
                }

                if collision {
                    penalty += min_overlap * min_overlap;
                }
            }
        }

        penalty
    }
}



impl<'a, 'b> CostFunction for BHProblem<'a, 'b> {
    type Param = Vec<FloatType>;
    type Output = FloatType;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.bh_function(param.as_slice()))
    }
}

impl<'a, 'b> Gradient for BHProblem<'a, 'b> {
    type Param = Vec<FloatType>;
    type Gradient = Vec<FloatType>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let n = param.len();
        let mut grad = vec![0.0 as FloatType; n];
        let eps = FloatType::EPSILON.sqrt();
        let mut xp = param.clone();
        let mut xm = param.clone();
        for i in 0..n {
            let xi = param[i];
            let h = eps * xi.abs().max(1.0 as FloatType);
            xp[i] = xi + h;
            xm[i] = xi - h;
            let fp = self.bh_function(xp.as_slice());
            let fm = self.bh_function(xm.as_slice());
            grad[i] = (fp - fm) / (2.0 as FloatType * h);
            xp[i] = xi;
            xm[i] = xi;
        }
        Ok(grad)
    }
}

fn initial_simplex(x0: &Vec<FloatType>) -> Vec<Vec<FloatType>> {
    let n = x0.len();
    let mut simplex: Vec<Vec<FloatType>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut xi = x0.clone();
        let delta = (0.05 as FloatType) * (1.0 as FloatType + x0[i].abs());
        xi[i] += delta;
        simplex.push(xi);
    }
    simplex
}

// def repetition(seed):
fn repetition(
    seed: usize,
    N: usize,
    nsi: usize,
    nsc: usize,
    penalty_tolerance: FloatType,
    final_step_size: FloatType,
    unit_polygon_vertices: &Array2<FloatType>,
    unit_polygon_vectors: &Array2<FloatType>,
    unit_container_vectors: &Array2<FloatType>,
    unit_container_apothem: FloatType,
) -> (FloatType, Vec<FloatType>) {
    println!("Attempt {}", seed);

    let mut rng = PcgInnerState64::mcg_seeded(seed as _);
    let rand01 = |rng: &mut PcgInnerState64| {
        let hi = rng.mcg_xsh_rs();
        let lo = rng.mcg_xsh_rs();
        let rand = ((hi as u64) << 32) | lo as u64;
        rand as FloatType / (u64::MAX as FloatType)
    };
    let mut dynamic_S =
        (N as FloatType).sqrt() * (2.0 as FloatType + rand01(&mut rng) * 2.0 as FloatType);
    let initial_S = dynamic_S;

    let mut x0 = vec![0.0 as FloatType; N * 3];

    if rand01(&mut rng) < 0.5 {
        for i in 0..(N * 3) {
            x0[i] = (rand01(&mut rng) - 0.5 as FloatType) * dynamic_S;
        }
    } else {
        let grid_count = ((N as FloatType).sqrt().ceil()) as usize;
        let min = -dynamic_S / (2.0 as FloatType) * (0.9 as FloatType);
        let max = dynamic_S / (2.0 as FloatType) * (0.9 as FloatType);
        let mut grid_points: Vec<(FloatType, FloatType)> =
            Vec::with_capacity(grid_count * grid_count);
        for gy in 0..grid_count {
            for gx in 0..grid_count {
                let xi = if grid_count > 1 {
                    min + (gx as FloatType) * ((max - min) / ((grid_count - 1) as FloatType))
                } else {
                    (min + max) / (2.0 as FloatType)
                };
                let yi = if grid_count > 1 {
                    min + (gy as FloatType) * ((max - min) / ((grid_count - 1) as FloatType))
                } else {
                    (min + max) / (2.0 as FloatType)
                };
                grid_points.push((xi, yi));
            }
        }
        for i in 0..N {
            let (gx, gy) = grid_points[i % grid_points.len()];
            x0[i * 3] = gx;
            x0[i * 3 + 1] = gy;
            x0[i * 3 + 2] = rand01(&mut rng) * (2.0 as FloatType) * PI;
        }
    }

    let mut last_valid_x = x0.clone();
    let mut last_valid_S = dynamic_S;

    // Match Python's local-minimizer `tol=1e-8` used in `minimize(..., tol=1e-8)`
    let solver_tol: FloatType = 1e-8 as FloatType;

    // Single arena reused for all BHProblem evaluations in this repetition.
    let arena = Bump::new();
    // Reusable arena-backed scratch buffers (wrapped in RefCell for interior mutability)
    let polys_x = Rc::new(RefCell::new(BumpVec::new_in(&arena)));
    let polys_y = Rc::new(RefCell::new(BumpVec::new_in(&arena)));
    let vecs_x = Rc::new(RefCell::new(BumpVec::new_in(&arena)));
    let vecs_y = Rc::new(RefCell::new(BumpVec::new_in(&arena)));

    loop {
        // Local minimization using L-BFGS (match Python's L-BFGS-B)
        let problem = BHProblem {
            N,
            nsi,
            nsc,
            unit_polygon_vertices,
            unit_polygon_vectors,
            unit_container_vectors,
            unit_container_apothem,
            S: dynamic_S,
            arena: &arena,
            polys_x: polys_x.clone(),
            polys_y: polys_y.clone(),
            vecs_x: vecs_x.clone(),
            vecs_y: vecs_y.clone(),
        };

        let armijo = ArmijoCondition::<FloatType>::new(0.0001).unwrap();
        let linesearch = BacktrackingLineSearch::new(armijo);
        let lbfgs = LBFGS::new(linesearch, 3)
            .with_tolerance_grad(solver_tol)
            .unwrap()
            .with_tolerance_cost(solver_tol)
            .unwrap();

        let result = Executor::new(problem, lbfgs)
            .configure(|state| state.param(x0.clone()))
            .run();

        let mut success = false;
        if let Ok(res) = result {
            let fun = res.state.get_best_cost();
            if fun < penalty_tolerance {
                if let Some(param) = res.state.get_best_param() {
                    last_valid_x = param.clone();
                    last_valid_S = dynamic_S;
                    success = true;
                    // compute multiplier (match Python exactly)
                    let base = (N as FloatType).sqrt() * (nsi as FloatType) / (nsc as FloatType);
                    let mut multiplier = 0.9999 as FloatType
                        - (dynamic_S - base) * (0.0099 as FloatType) / (initial_S - base);
                    multiplier = 1.0 as FloatType
                        - final_step_size
                        - (dynamic_S - base)
                            * ((0.01 as FloatType - final_step_size) / (initial_S - base));
                    for i in 0..x0.len() {
                        x0[i] = param[i] * multiplier;
                    }
                    dynamic_S *= multiplier;
                }
            } else {
                // Basinhopping-like global search (many local starts)
                let mut best_bh_fun = fun;
                let mut best_bh_param = res.state.get_best_param().cloned().unwrap_or(x0.clone());
                let mut current_fun = fun;
                let mut current_param = res.state.get_best_param().cloned().unwrap_or(x0.clone());

                for _ in 0..50 {
                    let mut trial = current_param.clone();
                    for k in 0..trial.len() {
                        trial[k] += (rand01(&mut rng) - 0.5 as FloatType)
                            * (2.0 as FloatType)
                            * (0.1 as FloatType);
                    }
                    let problem = BHProblem {
                        N,
                        nsi,
                        nsc,
                        unit_polygon_vertices,
                        unit_polygon_vectors,
                        unit_container_vectors,
                        unit_container_apothem,
                        S: dynamic_S,
                        arena: &arena,
                        polys_x: polys_x.clone(),
                        polys_y: polys_y.clone(),
                        vecs_x: vecs_x.clone(),
                        vecs_y: vecs_y.clone(),
                    };
                    let armijo = ArmijoCondition::<FloatType>::new(0.0001).unwrap();
                    let linesearch = BacktrackingLineSearch::new(armijo);
                    let lbfgs = LBFGS::new(linesearch, 3)
                        .with_tolerance_grad(solver_tol)
                        .unwrap()
                        .with_tolerance_cost(solver_tol)
                        .unwrap();
                    if let Ok(res2) = Executor::new(problem, lbfgs)
                        .configure(|state| state.param(trial.clone()))
                        .run()
                    {
                        let f2 = res2.state.get_best_cost();
                        if f2 < best_bh_fun {
                            best_bh_fun = f2;
                            best_bh_param = res2
                                .state
                                .get_best_param()
                                .cloned()
                                .unwrap_or(trial.clone());
                        }
                        // Metropolis acceptance
                        if f2 < current_fun
                            || ((current_fun - f2) / (0.1 as FloatType)).exp() > rand01(&mut rng)
                        {
                            current_fun = f2;
                            current_param = res2
                                .state
                                .get_best_param()
                                .cloned()
                                .unwrap_or(trial.clone());
                        }
                    }
                }

                if best_bh_fun < penalty_tolerance {
                    last_valid_x = best_bh_param.clone();
                    last_valid_S = dynamic_S;
                    let base = (N as FloatType).sqrt() * (nsi as FloatType) / (nsc as FloatType);
                    let mut multiplier = 0.9999 as FloatType
                        - (dynamic_S - base) * (0.0099 as FloatType) / (initial_S - base);
                    multiplier = 1.0 as FloatType
                        - final_step_size
                        - (dynamic_S - base)
                            * ((0.01 as FloatType - final_step_size) / (initial_S - base));
                    for i in 0..x0.len() {
                        x0[i] = last_valid_x[i] * multiplier;
                    }
                    dynamic_S *= multiplier;
                    success = true;
                } else {
                    break;
                }
            }
        } else {
            // If local solver failed for any reason, try a few random local starts
            let mut found = false;
            for _ in 0..10 {
                let mut trial = x0.clone();
                for k in 0..trial.len() {
                    trial[k] +=
                        (rand01(&mut rng) - 0.5 as FloatType) * dynamic_S * 0.1 as FloatType;
                }
                let problem = BHProblem {
                    N,
                    nsi,
                    nsc,
                    unit_polygon_vertices,
                    unit_polygon_vectors,
                    unit_container_vectors,
                    unit_container_apothem,
                    S: dynamic_S,
                    arena: &arena,
                    polys_x: polys_x.clone(),
                    polys_y: polys_y.clone(),
                    vecs_x: vecs_x.clone(),
                    vecs_y: vecs_y.clone(),
                };
                let armijo = ArmijoCondition::<FloatType>::new(0.0001).unwrap();
                let linesearch = BacktrackingLineSearch::new(armijo);
                let lbfgs = LBFGS::new(linesearch, 3)
                    .with_tolerance_grad(solver_tol)
                    .unwrap()
                    .with_tolerance_cost(solver_tol)
                    .unwrap();
                if let Ok(res2) = Executor::new(problem, lbfgs)
                    .configure(|state| state.param(trial.clone()))
                    .run()
                {
                    let f2 = res2.state.get_best_cost();
                    if f2 < penalty_tolerance {
                        last_valid_x = res2
                            .state
                            .get_best_param()
                            .cloned()
                            .unwrap_or(trial.clone());
                        last_valid_S = dynamic_S;
                        x0 = last_valid_x.clone();
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break;
            }
        }

        if !success {
            break;
        }
    }

    (last_valid_S, last_valid_x)
}

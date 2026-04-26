// changing `a * b + c` to `a.mul_add(b, c)` literally leads to more
// verification errors, so we don't do that.
#![allow(clippy::suboptimal_flops)]
use std::f64::consts::PI;
use std::iter;
use std::time::Duration;

use bumpalo::Bump;
use clap::Parser;
use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;
use plotters::prelude::*;
use rayon::prelude::*;
use voxell_rng::rng::pcg_advanced::pcg_64::PcgInnerState64;
use voxell_timer::time_fn;

const MAX_ITERATIONS: usize = 5000;
const BH_STEPS: usize = 50;
const BH_STEP_SIZE: f64 = 0.2;
const BH_TEMP: f64 = 0.1;
const VERIFY_EPS: f64 = 1e-9;
const GRADIENT_EPS_REL: f64 = 1.0;

#[derive(Parser, Debug, Clone)]
pub struct Args {
    /// Number of inner polygons
    pub inner_polygons: usize,
    /// Number of sides of the inner polygons
    pub inner_sides: usize,
    /// Number of sides of the container polygon
    pub container_sides: usize,
    /// Number of attempts to run
    #[arg(long, default_value_t = 1000)]
    pub attempts: usize,
    /// Overlap penalty tolerance. Probably best left at default.
    #[arg(long, default_value_t = 1e-8)]
    pub tolerance: f64,
    /// How small the last theoretical step in container size decrease will be
    /// (it gets smaller over time)
    #[arg(long, default_value_t = 0.0001)]
    pub finalstep: f64,
}

fn main() {
    eprintln!(
        "Polygon Packer - Packing regular polygons into a regular container"
    );
    let args = Args::parse();
    let geo = Geometry::new(args.inner_sides, args.container_sides);

    println!(
        "Starting packing with {} polygons ({} sides each) into a {}-sided container",
        args.inner_polygons, args.inner_sides, args.container_sides
    );
    let (results, duration): (Vec<(f64, Vec<f64>)>, Duration) = time_fn(|| {
        (0..args.attempts)
            .into_par_iter()
            .map(|i| repetition(i, &args, &geo))
            .collect()
    });

    println!("Completed {} attempts in {:.2?}", args.attempts, duration);

    if let Some((best_s, best_v)) = results
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    {
        let side_len = best_s * (PI / args.container_sides as f64).sin()
            / (PI / args.inner_sides as f64).sin();
        println!("Final side length: {}", side_len);

        verify_results(&best_v, best_s, &args, &geo);
        render(&args, &geo, best_s, &best_v);
    }
}

struct Geometry {
    inner_verts_x: Vec<f64>,
    inner_verts_y: Vec<f64>,
    inner_axes_x: Vec<f64>,
    inner_axes_y: Vec<f64>,
    cont_axes_x: Vec<f64>,
    cont_axes_y: Vec<f64>,
    apothem: f64,
}

impl Geometry {
    fn new(nsi: usize, nsc: usize) -> Self {
        let i_angle = 2.0 * PI / nsi as f64;
        let c_angle = 2.0 * PI / nsc as f64;

        let mut inner_verts_x = Vec::with_capacity(nsi);
        let mut inner_verts_y = Vec::with_capacity(nsi);
        for i in 0..nsi {
            inner_verts_x.push((i as f64 * i_angle).cos());
            inner_verts_y.push((i as f64 * i_angle).sin());
        }

        let mut inner_axes_x = Vec::with_capacity(nsi);
        let mut inner_axes_y = Vec::with_capacity(nsi);
        for i in 0..nsi {
            let a = i as f64 * i_angle + PI / nsi as f64;
            inner_axes_x.push(a.cos());
            inner_axes_y.push(a.sin());
        }

        let mut cont_axes_x = Vec::with_capacity(nsc);
        let mut cont_axes_y = Vec::with_capacity(nsc);
        for i in 0..nsc {
            let a = i as f64 * c_angle + PI / nsc as f64;
            cont_axes_x.push(a.cos());
            cont_axes_y.push(a.sin());
        }

        Self {
            inner_verts_x,
            inner_verts_y,
            inner_axes_x,
            inner_axes_y,
            cont_axes_x,
            cont_axes_y,
            apothem: (PI / nsc as f64).cos(),
        }
    }
}

struct ScratchPad<'a> {
    // (x, y) vertices for each polygon,
    // flattened: [poly0_vx0, poly0_vy0, poly0_vx1, ...]
    poly_verts: &'a mut [f64],
    // (x, y) axes for each polygon, flattened
    poly_axes: &'a mut [f64],
}

impl<'a> ScratchPad<'a> {
    fn new(bump: &'a Bump, n_polys: usize, n_sides: usize) -> Self {
        let v_size = n_polys * n_sides * 2;
        let a_size = n_polys * n_sides * 2;
        Self {
            poly_verts: bump.alloc_slice_fill_copy(v_size, 0.0),
            poly_axes: bump.alloc_slice_fill_copy(a_size, 0.0),
        }
    }
}

fn penalty(
    v: &[f64], s: f64, args: &Args, geo: &Geometry, pad: &mut ScratchPad,
) -> f64 {
    let mut p = 0.0;
    let limit = geo.apothem * s;
    let nsi = args.inner_sides;

    for i in 0..args.inner_polygons {
        let (px, py, pr) = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
        let (sin_r, cos_r) = pr.sin_cos();

        for j in 0..nsi {
            let vx = geo.inner_verts_x[j];
            let vy = geo.inner_verts_y[j];
            let tx = cos_r * vx - sin_r * vy + px;
            let ty = sin_r * vx + cos_r * vy + py;

            pad.poly_verts[(i * nsi + j) * 2] = tx;
            pad.poly_verts[(i * nsi + j) * 2 + 1] = ty;

            for k in 0..geo.cont_axes_x.len() {
                let dist = tx * geo.cont_axes_x[k] + ty * geo.cont_axes_y[k];
                if dist > limit {
                    p += (dist - limit).powi(2);
                }
            }
        }

        for j in 0..nsi {
            let ax = geo.inner_axes_x[j];
            let ay = geo.inner_axes_y[j];
            pad.poly_axes[(i * nsi + j) * 2] = cos_r * ax - sin_r * ay;
            pad.poly_axes[(i * nsi + j) * 2 + 1] = sin_r * ax + cos_r * ay;
        }
    }

    for i in 0..args.inner_polygons {
        let i_off = i * nsi * 2;
        for j in i + 1..args.inner_polygons {
            let j_off = j * nsi * 2;
            let mut min_overlap = 1e20;
            let mut collision = true;

            for k in 0..(nsi * 2) {
                let (ax, ay) = if k < nsi {
                    (
                        pad.poly_axes[i_off + k * 2],
                        pad.poly_axes[i_off + k * 2 + 1],
                    )
                } else {
                    let k2 = k - nsi;
                    (
                        pad.poly_axes[j_off + k2 * 2],
                        pad.poly_axes[j_off + k2 * 2 + 1],
                    )
                };

                let (min1, max1) =
                    project_soa(pad.poly_verts, i_off, nsi, ax, ay);
                let (min2, max2) =
                    project_soa(pad.poly_verts, j_off, nsi, ax, ay);

                let overlap = max1.min(max2) - min1.max(min2);
                if overlap <= 0.0 {
                    collision = false;
                    break;
                }
                if overlap < min_overlap {
                    min_overlap = overlap;
                }
            }

            if collision {
                p += min_overlap * min_overlap;
            }
        }
    }
    p
}

#[inline(always)]
fn project_soa(
    verts: &[f64], offset: usize, n: usize, ax: f64, ay: f64,
) -> (f64, f64) {
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    for i in 0..n {
        let dot = verts[offset + i * 2] * ax + verts[offset + i * 2 + 1] * ay;
        if dot < min {
            min = dot;
        }
        if dot > max {
            max = dot;
        }
    }
    (min, max)
}

fn local_min(
    x0: &[f64], s: f64, args: &Args, geo: &Geometry, bump: &mut Bump,
) -> Option<Vec<f64>> {
    let mut x = x0.to_vec();
    bump.reset();

    struct OptCtx<'a> {
        pad: ScratchPad<'a>,
        x_tmp: Vec<f64>,
    }

    let ctx = OptCtx {
        pad: ScratchPad::new(&*bump, args.inner_polygons, args.inner_sides),
        x_tmp: x0.to_vec(),
    };

    let obj = |xx: &[f64], grad: Option<&mut [f64]>, ctx: &mut OptCtx| -> f64 {
        let f0 = penalty(xx, s, args, geo, &mut ctx.pad);
        if let Some(g) = grad {
            let eps = f64::EPSILON.sqrt();
            ctx.x_tmp.copy_from_slice(xx);
            for i in 0..xx.len() {
                let step = eps * (GRADIENT_EPS_REL + xx[i].abs());
                ctx.x_tmp[i] += step;
                let f1 = penalty(&ctx.x_tmp, s, args, geo, &mut ctx.pad);
                g[i] = (f1 - f0) / step;
                ctx.x_tmp[i] = xx[i];
            }
        }
        f0
    };

    let mut opt =
        Nlopt::new(Algorithm::Lbfgs, x0.len(), obj, Target::Minimize, ctx);

    opt.set_ftol_rel(args.tolerance).ok();
    opt.set_xtol_rel(args.tolerance).ok();
    opt.set_maxeval(MAX_ITERATIONS as _).ok();

    match opt.optimize(&mut x) {
        Ok(_) => Some(x),
        Err(_) => None,
    }
}
fn repetition(seed: usize, args: &Args, geo: &Geometry) -> (f64, Vec<f64>) {
    let mut rng = PcgInnerState64::mcg_seeded(seed as u64);
    eprintln!(
        "Starting attempt {} with seed {}",
        seed,
        rng.oneseq_rxs_m_xs()
    );
    let mut rand = || rng.oneseq_rxs_m_xs() as f64 / u64::MAX as f64;

    let mut s = (args.inner_polygons as f64).sqrt() * (2.0 + rand() * 2.0);
    let initial_s = s;
    let mut x = vec![0.0; args.inner_polygons * 3];

    if rand() < 0.5 {
        x.iter_mut().for_each(|v| *v = (rand() - 0.5) * s);
    } else {
        let gc = (args.inner_polygons as f64).sqrt().ceil() as usize;
        let span = s * 0.45;
        for i in 0..args.inner_polygons {
            let step = if gc > 1 {
                2.0 * span / (gc - 1) as f64
            } else {
                0.0
            };
            x[i * 3] = (i % gc) as f64 * step - span;
            x[i * 3 + 1] = (i / gc) as f64 * step - span;
            x[i * 3 + 2] = rand() * 2.0 * PI;
        }
    }

    let mut bump = Bump::new();
    let (mut last_x, mut last_s) = (x.clone(), s);

    loop {
        if let Some(refined) = local_min(&x, s, args, geo, &mut bump) {
            let mut pad =
                ScratchPad::new(&bump, args.inner_polygons, args.inner_sides);
            let p = penalty(&refined, s, args, geo, &mut pad);
            if p < args.tolerance {
                last_x.clone_from(&refined);
                last_s = s;
                let base = (args.inner_polygons as f64).sqrt()
                    * args.inner_sides as f64
                    / args.container_sides as f64;
                let m = (s - base).mul_add(
                    -((0.01 - args.finalstep) / (initial_s - base)),
                    1.0 - args.finalstep,
                );
                for i in 0..args.inner_polygons {
                    x[i * 3] = refined[i * 3] * m;
                    x[i * 3 + 1] = refined[i * 3 + 1] * m;
                    x[i * 3 + 2] = refined[i * 3 + 2];
                }
                s *= m;
                continue;
            }
            if let Some(bh) = run_bh(&refined, p, s, args, geo, &mut rand) {
                last_x.clone_from(&bh);
                last_s = s;
                let base = (args.inner_polygons as f64).sqrt()
                    * args.inner_sides as f64
                    / args.container_sides as f64;
                let m = (s - base).mul_add(
                    -((0.01 - args.finalstep) / (initial_s - base)),
                    1.0 - args.finalstep,
                );
                for i in 0..args.inner_polygons {
                    x[i * 3] = bh[i * 3] * m;
                    x[i * 3 + 1] = bh[i * 3 + 1] * m;
                    x[i * 3 + 2] = bh[i * 3 + 2];
                }
                s *= m;
                continue;
            }
        }
        break;
    }
    (last_s, last_x)
}

fn run_bh(
    refined: &[f64], p: f64, s: f64, args: &Args, geo: &Geometry,
    rand: &mut dyn FnMut() -> f64,
) -> Option<Vec<f64>> {
    let (mut best_p, mut best_x) = (p, refined.to_vec());
    let (mut cur_p, mut cur_x) = (p, refined.to_vec());
    let mut bump = Bump::new();

    for _ in 0..BH_STEPS {
        let trial: Vec<_> = cur_x
            .iter()
            .map(|&v| v + (rand() - 0.5) * BH_STEP_SIZE)
            .collect();
        if let Some(opt_x) = local_min(&trial, s, args, geo, &mut bump) {
            let mut pad =
                ScratchPad::new(&bump, args.inner_polygons, args.inner_sides);
            let opt_p = penalty(&opt_x, s, args, geo, &mut pad);
            if opt_p < best_p {
                best_p = opt_p;
                best_x.clone_from(&opt_x);
            }
            if opt_p < cur_p || ((cur_p - opt_p) / BH_TEMP).exp() > rand() {
                cur_p = opt_p;
                cur_x = opt_x;
            }
            if best_p < args.tolerance {
                return Some(best_x);
            }
        }
    }
    None
}

fn verify_results(v: &[f64], s: f64, args: &Args, geo: &Geometry) {
    println!("--- Verification Pass ---");
    let bump = Bump::new();
    let pad = ScratchPad::new(&bump, args.inner_polygons, args.inner_sides);
    let limit = geo.apothem * s;
    let mut overlap_count = 0;
    let mut out_of_bounds = 0;
    let nsi = args.inner_sides;

    for i in 0..args.inner_polygons {
        let (px, py, pr) = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
        let (sin_r, cos_r) = pr.sin_cos();
        for j in 0..nsi {
            let tx = cos_r * geo.inner_verts_x[j]
                - sin_r * geo.inner_verts_y[j]
                + px;
            let ty = sin_r * geo.inner_verts_x[j]
                + cos_r * geo.inner_verts_y[j]
                + py;
            pad.poly_verts[(i * nsi + j) * 2] = tx;
            pad.poly_verts[(i * nsi + j) * 2 + 1] = ty;
        }
        for j in 0..nsi {
            let ax = geo.inner_axes_x[j];
            let ay = geo.inner_axes_y[j];
            pad.poly_axes[(i * nsi + j) * 2] = cos_r * ax - sin_r * ay;
            pad.poly_axes[(i * nsi + j) * 2 + 1] = sin_r * ax + cos_r * ay;
        }
    }

    for i in 0..args.inner_polygons {
        let mut is_out = false;
        for j in 0..nsi {
            let tx = pad.poly_verts[(i * nsi + j) * 2];
            let ty = pad.poly_verts[(i * nsi + j) * 2 + 1];
            if geo
                .cont_axes_x
                .iter()
                .zip(&geo.cont_axes_y)
                .any(|(&ax, &ay)| tx * ax + ty * ay > limit + VERIFY_EPS)
            {
                is_out = true;
                break;
            }
        }
        if is_out {
            out_of_bounds += 1;
        }

        for j in i + 1..args.inner_polygons {
            let mut collision = true;
            for k in 0..(nsi * 2) {
                let (ax, ay) = if k < nsi {
                    (
                        pad.poly_axes[(i * nsi + k) * 2],
                        pad.poly_axes[(i * nsi + k) * 2 + 1],
                    )
                } else {
                    let k2 = k - nsi;
                    (
                        pad.poly_axes[(j * nsi + k2) * 2],
                        pad.poly_axes[(j * nsi + k2) * 2 + 1],
                    )
                };
                let (min1, max1) =
                    project_soa(pad.poly_verts, i * nsi * 2, nsi, ax, ay);
                let (min2, max2) =
                    project_soa(pad.poly_verts, j * nsi * 2, nsi, ax, ay);
                if max1.min(max2) - min1.max(min2) <= VERIFY_EPS {
                    collision = false;
                    break;
                }
            }
            if collision {
                overlap_count += 1;
            }
        }
    }
    println!(
        "Out of bounds: {}, Pairwise overlaps: {}",
        out_of_bounds, overlap_count
    );
}

fn render(args: &Args, geo: &Geometry, s: f64, v: &[f64]) {
    let path = format!(
        "{}_{}_in_{}.png",
        args.inner_polygons, args.inner_sides, args.container_sides
    );
    let root = BitMapBackend::new(&path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE).ok();
    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_2d(-s..s, -s..s)
        .unwrap();

    let mut c_pts: Vec<_> = (0..args.container_sides)
        .map(|i| {
            let a = i as f64 * (2.0 * PI / args.container_sides as f64);
            (a.cos() * s, a.sin() * s)
        })
        .collect();
    c_pts.push(c_pts[0]);
    chart
        .draw_series(iter::once(PathElement::new(c_pts, BLACK)))
        .ok();

    for i in 0..args.inner_polygons {
        let (px, py, pr) = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
        let (sin_r, cos_r) = pr.sin_cos();
        let mut p_pts: Vec<_> = (0..args.inner_sides)
            .map(|j| {
                let tx = cos_r * geo.inner_verts_x[j]
                    - sin_r * geo.inner_verts_y[j]
                    + px;
                let ty = sin_r * geo.inner_verts_x[j]
                    + cos_r * geo.inner_verts_y[j]
                    + py;
                (tx, ty)
            })
            .collect();
        chart
            .draw_series(iter::once(Polygon::new(
                p_pts.clone(),
                RGBColor(204, 204, 204).filled(),
            )))
            .ok();
        p_pts.push(p_pts[0]);
        chart
            .draw_series(iter::once(PathElement::new(p_pts, BLACK)))
            .ok();
    }
}

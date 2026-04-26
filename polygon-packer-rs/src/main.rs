use std::f64::consts::PI;

use clap::Parser;
use nalgebra::Rotation2;
use nalgebra::Vector2;
use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;
use plotters::prelude::*;
use rayon::prelude::*;
use voxell_rng::rng::pcg_advanced::pcg_64::PcgInnerState64;

fn main() {
    let args = Args::parse();
    let geo = Geometry::new(args.inner_sides, args.container_sides);

    let results: Vec<(f64, Vec<f64>)> = (0..args.attempts)
        .into_par_iter()
        .map(|i| repetition(i, &args, &geo))
        .collect();

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

#[derive(Parser, Debug, Clone)]
pub struct Args {
    pub inner_polygons: usize,
    pub inner_sides: usize,
    pub container_sides: usize,
    #[arg(long, default_value_t = 1000)]
    pub attempts: usize,
    #[arg(long, default_value_t = 1e-8)]
    pub tolerance: f64,
    #[arg(long, default_value_t = 0.0001)]
    pub finalstep: f64,
}

struct Geometry {
    inner_verts: Vec<Vector2<f64>>,
    inner_axes: Vec<Vector2<f64>>,
    cont_axes: Vec<Vector2<f64>>,
    apothem: f64,
}

impl Geometry {
    fn new(nsi: usize, nsc: usize) -> Self {
        let i_angle = 2.0 * PI / nsi as f64;
        let c_angle = 2.0 * PI / nsc as f64;
        let inner_verts = (0..nsi)
            .map(|i| {
                Vector2::new(
                    (i as f64 * i_angle).cos(),
                    (i as f64 * i_angle).sin(),
                )
            })
            .collect();
        let inner_axes = (0..nsi)
            .map(|i| {
                let a = i as f64 * i_angle + PI / nsi as f64;
                Vector2::new(a.cos(), a.sin())
            })
            .collect();
        let cont_axes = (0..nsc)
            .map(|i| {
                let a = i as f64 * c_angle + PI / nsc as f64;
                Vector2::new(a.cos(), a.sin())
            })
            .collect();
        Self {
            inner_verts,
            inner_axes,
            cont_axes,
            apothem: (PI / nsc as f64).cos(),
        }
    }
}

fn penalty(v: &[f64], s: f64, args: &Args, geo: &Geometry) -> f64 {
    let mut p = 0.0;
    let limit = geo.apothem * s;

    // 1. Container Penalty (Poking)
    for i in 0..args.inner_polygons {
        let (pos, rot) = (
            Vector2::new(v[i * 3], v[i * 3 + 1]),
            Rotation2::new(v[i * 3 + 2]),
        );
        for iv in &geo.inner_verts {
            let pt = rot * iv + pos;
            for ax in &geo.cont_axes {
                let dist = pt.dot(ax);
                if dist > limit {
                    p += (dist - limit).powi(2);
                }
            }
        }
    }

    // 2. Pairwise Collision Penalty (SAT)
    let polys: Vec<_> = (0..args.inner_polygons)
        .map(|i| {
            let (pos, rot) = (
                Vector2::new(v[i * 3], v[i * 3 + 1]),
                Rotation2::new(v[i * 3 + 2]),
            );
            (
                geo.inner_verts
                    .iter()
                    .map(|&iv| rot * iv + pos)
                    .collect::<Vec<_>>(),
                geo.inner_axes
                    .iter()
                    .map(|&ia| rot * ia)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    for i in 0..args.inner_polygons {
        for j in i + 1..args.inner_polygons {
            let mut min_overlap = 1e20;
            let mut collision = true;
            for ax in polys[i].1.iter().chain(polys[j].1.iter()) {
                let (min1, max1) = project(&polys[i].0, ax);
                let (min2, max2) = project(&polys[j].0, ax);
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

fn project(verts: &[Vector2<f64>], ax: &Vector2<f64>) -> (f64, f64) {
    verts
        .iter()
        .map(|v| v.dot(ax))
        .fold((f64::MAX, f64::MIN), |(a, b), d| (a.min(d), b.max(d)))
}

/// Matches SciPy's numerical gradient calculation (Forward Difference)
fn local_min(
    x0: &[f64], s: f64, args: &Args, geo: &Geometry,
) -> Option<Vec<f64>> {
    let mut x = x0.to_vec();
    let obj = |xx: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let f0 = penalty(xx, s, args, geo);
        if let Some(g) = grad {
            // SciPy uses a small epsilon relative to the scale of the value
            let eps = f64::EPSILON.sqrt();
            let mut x_tmp = xx.to_vec();
            for i in 0..xx.len() {
                let step = eps * (1.0 + xx[i].abs());
                x_tmp[i] += step;
                let f1 = penalty(&x_tmp, s, args, geo);
                g[i] = (f1 - f0) / step;
                x_tmp[i] = xx[i]; // Reset for next dimension
            }
        }
        f0
    };

    let mut opt =
        Nlopt::new(Algorithm::Lbfgs, x0.len(), obj, Target::Minimize, ());

    // Mapping SciPy's 'tol' to NLopt tolerances
    opt.set_ftol_rel(args.tolerance).ok();
    opt.set_xtol_rel(args.tolerance).ok();
    opt.set_maxeval(5000).ok(); // Standard limit to prevent infinite loops

    match opt.optimize(&mut x) {
        Ok(_) => Some(x),
        Err(_) => None,
    }
}

fn verify_results(v: &[f64], s: f64, args: &Args, geo: &Geometry) {
    println!("--- Verification Pass ---");
    let limit = geo.apothem * s;
    let mut overlap_count = 0;
    let mut out_of_bounds = 0;

    let polys: Vec<_> = (0..args.inner_polygons)
        .map(|i| {
            let (pos, rot) = (
                Vector2::new(v[i * 3], v[i * 3 + 1]),
                Rotation2::new(v[i * 3 + 2]),
            );
            (
                geo.inner_verts
                    .iter()
                    .map(|&iv| rot * iv + pos)
                    .collect::<Vec<_>>(),
                geo.inner_axes
                    .iter()
                    .map(|&ia| rot * ia)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    for (i, verts) in polys.iter().map(|x| &x.0).enumerate() {
        if verts
            .iter()
            .any(|pt| geo.cont_axes.iter().any(|ax| pt.dot(ax) > limit + 1e-9))
        {
            out_of_bounds += 1;
        }
        for (j, ()) in polys.iter().map(|_x| ()).enumerate().skip(i + 1) {
            let mut collision = true;
            for ax in polys[i].1.iter().chain(polys[j].1.iter()) {
                let (min1, max1) = project(&polys[i].0, ax);
                let (min2, max2) = project(&polys[j].0, ax);
                if max1.min(max2) - min1.max(min2) <= 1e-9 {
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

fn repetition(seed: usize, args: &Args, geo: &Geometry) -> (f64, Vec<f64>) {
    let mut rng = PcgInnerState64::mcg_seeded(seed as u64);
    let mut rand = || {
        (((rng.mcg_xsh_rs() as u64) << 32) | rng.mcg_xsh_rs() as u64) as f64
            / u64::MAX as f64
    };
    let mut s = (args.inner_polygons as f64).sqrt() * (2.0 + rand() * 2.0);
    let initial_s = s;
    let mut x = vec![0.0; args.inner_polygons * 3];

    if rand() < 0.5 {
        x.iter_mut().for_each(|v| *v = (rand() - 0.5) * s);
    } else {
        let gc = (args.inner_polygons as f64).sqrt().ceil() as usize;
        let span = s * 0.45;
        for i in 0..args.inner_polygons {
            x[i * 3] = if gc > 1 {
                (i % gc) as f64 * (2.0 * span / (gc - 1) as f64) - span
            } else {
                0.0
            };
            x[i * 3 + 1] = if gc > 1 {
                (i / gc) as f64 * (2.0 * span / (gc - 1) as f64) - span
            } else {
                0.0
            };
            x[i * 3 + 2] = rand() * 2.0 * PI;
        }
    }

    let (mut last_x, mut last_s) = (x.clone(), s);
    loop {
        if let Some(refined) = local_min(&x, s, args, geo) {
            let p = penalty(&refined, s, args, geo);
            if p < args.tolerance {
                last_x = refined.clone();
                last_s = s;
                let base = (args.inner_polygons as f64).sqrt()
                    * args.inner_sides as f64
                    / args.container_sides as f64;
                let m = (s - base).mul_add(
                    -((0.01 - args.finalstep) / (initial_s - base)),
                    1.0 - args.finalstep,
                );
                x = refined.iter().map(|v| v * m).collect();
                s *= m;
                continue;
            }
            if let Some(bh) = run_bh(&refined, p, s, args, geo, &mut rand) {
                last_x = bh.clone();
                last_s = s;
                let base = (args.inner_polygons as f64).sqrt()
                    * args.inner_sides as f64
                    / args.container_sides as f64;
                let m = (s - base).mul_add(
                    -((0.01 - args.finalstep) / (initial_s - base)),
                    1.0 - args.finalstep,
                );
                x = bh.iter().map(|v| v * m).collect();
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
    for _ in 0..50 {
        let trial: Vec<_> =
            cur_x.iter().map(|&v| v + (rand() - 0.5) * 0.2).collect();
        if let Some(opt_x) = local_min(&trial, s, args, geo) {
            let opt_p = penalty(&opt_x, s, args, geo);
            if opt_p < best_p {
                best_p = opt_p;
                best_x = opt_x.clone();
            }
            if opt_p < cur_p || ((cur_p - opt_p) / 0.1).exp() > rand() {
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
    c_pts.push(c_pts[0]); // Close outline
    chart
        .draw_series(std::iter::once(PathElement::new(c_pts, BLACK)))
        .ok();

    for i in 0..args.inner_polygons {
        let (pos, rot) = (
            Vector2::new(v[i * 3], v[i * 3 + 1]),
            Rotation2::new(v[i * 3 + 2]),
        );
        let mut p_pts: Vec<_> = geo
            .inner_verts
            .iter()
            .map(|&iv| {
                let p = rot * iv + pos;
                (p.x, p.y)
            })
            .collect();
        chart
            .draw_series(std::iter::once(Polygon::new(
                p_pts.clone(),
                RGBColor(204, 204, 204).filled(),
            )))
            .ok();
        p_pts.push(p_pts[0]); // Close outline
        chart
            .draw_series(std::iter::once(PathElement::new(p_pts, BLACK)))
            .ok();
    }
}

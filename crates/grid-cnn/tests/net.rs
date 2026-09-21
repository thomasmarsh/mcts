use grid_cnn::{cell_map, transform_planes, Net, Weights, SYMMETRIES};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

struct Fixture {
    net: Net,
    n: usize,
    planes: Vec<f32>,
    values: Vec<f32>,
    logits: Vec<f32>,
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn fixture() -> Fixture {
    let weights = Weights::load(format!("{DIR}/weights.bin")).unwrap();
    let io: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/io.json")).unwrap()).unwrap();
    Fixture {
        net: Net::new(&weights),
        n: io["n"].as_u64().unwrap() as usize,
        planes: floats(&io["planes"]),
        values: floats(&io["values"]),
        logits: floats(&io["logits"]),
    }
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}[{i}]: rust {g} vs torch {w}");
    }
}

#[test]
fn forward_matches_the_torch_reference() {
    let f = fixture();
    let out = f.net.forward(&f.planes, f.n);
    assert_close(&out.values, &f.values, 1e-4, "value");
    assert_close(&out.logits, &f.logits, 1e-4, "logits");
}

#[test]
fn a_batch_equals_its_positions_evaluated_alone() {
    let f = fixture();
    let g = f.net.geometry();
    let per = g.cells() * g.in_planes;
    let batch = f.net.forward(&f.planes, f.n);
    for i in 0..f.n {
        let one = f.net.forward(&f.planes[i * per..(i + 1) * per], 1);
        assert_close(&one.values, &batch.values[i..i + 1], 1e-5, "value");
        assert_close(
            &one.logits,
            &batch.logits[i * g.policy_out..(i + 1) * g.policy_out],
            1e-5,
            "logits",
        );
    }
    assert!(f.net.forward(&[], 0).values.is_empty());
}

/// Average over all 8 orientations, mapped back to the input frame: cell logits follow the cells,
/// the trailing (non-cell) logits and the value are frame independent.
fn ensemble(f: &Fixture, planes: &[f32]) -> (f32, Vec<f32>) {
    let g = f.net.geometry();
    let cells = g.cells();
    let (mut value, mut logits) = (0.0f32, vec![0.0f32; g.policy_out]);
    for sym in 0..SYMMETRIES {
        let out = f
            .net
            .forward(&transform_planes(planes, g.size, g.in_planes, sym), 1);
        let map = cell_map(g.size, sym);
        value += out.values[0] / SYMMETRIES as f32;
        for cell in 0..cells {
            logits[cell] += out.logits[map[cell]] / SYMMETRIES as f32;
        }
        for (acc, extra) in logits[cells..].iter_mut().zip(&out.logits[cells..]) {
            *acc += extra / SYMMETRIES as f32;
        }
    }
    (value, logits)
}

#[test]
fn the_orientation_ensemble_is_d4_equivariant() {
    let f = fixture();
    let g = f.net.geometry();
    let per = g.cells() * g.in_planes;
    let x = &f.planes[..per];
    let (v0, l0) = ensemble(&f, x);
    for sym in 1..SYMMETRIES {
        let (v, l) = ensemble(&f, &transform_planes(x, g.size, g.in_planes, sym));
        assert!(
            (v - v0).abs() < 1e-4,
            "value not invariant under sym {sym}: {v} vs {v0}"
        );
        let map = cell_map(g.size, sym);
        for cell in 0..g.cells() {
            assert!(
                (l[map[cell]] - l0[cell]).abs() < 1e-4,
                "cell logit not equivariant under sym {sym}"
            );
        }
        for extra in g.cells()..g.policy_out {
            assert!(
                (l[extra] - l0[extra]).abs() < 1e-4,
                "extra logit not invariant under sym {sym}"
            );
        }
    }
}

//! Reproduces the 64 x 64, 4000-iteration errors of
//! docs/runs/2026-10-05-jacobi-gates.md section 4 (accuracy.py, exp-jacobi).

use jacobi::bitsliced::Bitsliced;
use jacobi::reference::f32_run;
use jacobi::Problem;

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 0.0005 * want.abs()
}

#[test]
fn errors_match_the_run_doc() {
    let p = Problem::new(64).unwrap();
    let doc = [(4, 9.092e-01, 5.328e-01), (8, 6.680e-01, 5.016e-01), (12, 1.623e-01, 1.550e-02), (16, 1.264e-02, 3.396e-03)];
    for (bits, trunc, round) in doc {
        for (r, want) in [(false, trunc), (true, round)] {
            let mut e = Bitsliced::from_problem(&p, bits);
            e.round = r;
            e.run(4000);
            let got = p.max_error_q(&e.read_q(), bits);
            assert!(close(got, want), "b={bits} round={r}: {got:.4e}, doc {want:.3e}");
        }
    }
    // float64 in the doc: 5.592e-03; f32 lands within 1%.
    let mut f = p.initial_f32();
    f32_run(64, &mut f, 4000);
    let got = p.max_error_f32(&f);
    assert!((got - 5.592e-3).abs() < 0.01 * 5.592e-3, "f32: {got:.4e}");
}

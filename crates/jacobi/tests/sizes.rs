use jacobi::grid_size;

#[test]
fn sizes_round_up_or_fail_with_a_message() {
    assert_eq!(grid_size(512.0), Ok(512));
    assert_eq!(grid_size(500.0), Ok(512));
    assert_eq!(grid_size(1.0), Ok(64));
    assert_eq!(grid_size(100.5), Ok(128));
    assert_eq!(grid_size(4096.0), Ok(4096));
    assert!(grid_size(4097.0).is_err());
    assert!(grid_size(0.0).is_err());
    assert!(grid_size(-3.0).is_err());
    assert!(grid_size(f64::NAN).is_err());
    assert!(jacobi::Problem::new(100).is_err());
}

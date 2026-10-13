//! workflows' sparse-data decision (`workflows:::should_use_sparsity`, workflows
//! 1.3.0): before fitting, a workflow estimates how sparse the baked design will be
//! and, for engines that accept sparse matrices, tells `step_dummy()` to emit sparse
//! columns. parsnip then passes a `dgCMatrix` to the engine, which for xgboost means
//! **every exact zero of the design is a missing value** with a learned default
//! direction. The decision is a fitted piecewise-linear model in the estimated
//! sparsity and the number of rows; it is transcribed here coefficient by
//! coefficient so that a Rust workflow makes the same choice.

/// `workflows:::should_use_sparsity(sparsity, engine, n_rows) == "yes"`.
pub fn should_use_sparsity(sparsity: f64, engine: &str, n_rows: usize) -> bool {
    if engine == "ranger" {
        return false;
    }
    let n_rows = n_rows as f64;
    let k = 0.864864864864865;
    let lo = if sparsity < k { k - sparsity } else { 0.0 };
    let hi = if sparsity > k { sparsity - k } else { 0.0 };
    let r16_lo = if n_rows < 16000.0 { 16000.0 - n_rows } else { 0.0 };
    let r16_hi = if n_rows > 16000.0 { n_rows - 16000.0 } else { 0.0 };
    let r8_lo = if n_rows < 8000.0 { 8000.0 - n_rows } else { 0.0 };
    let r8_hi = if n_rows > 8000.0 { n_rows - 8000.0 } else { 0.0 };
    let mut log_fold = -0.104969175543923
        + (lo * -0.863987224202949)
        + (lo * r16_lo * 7.24704434279721e-05)
        + (lo * r16_hi * -9.34342073897278e-07)
        + (hi * 9.28694536071564)
        + (hi * r8_lo * -0.000439058585009898)
        + (hi * r8_hi * 2.66853007695264e-05);
    if engine == "xgboost" {
        let k2 = 0.984615384615385;
        let lo2 = if sparsity < k2 { k2 - sparsity } else { 0.0 };
        let hi2 = if sparsity > k2 { sparsity - k2 } else { 0.0 };
        log_fold += (lo2 * -0.558117285856414)
            + (r8_lo * -0.000104996462576298)
            + (r8_hi * 2.36552247695297e-06)
            + (hi2 * 40.8016066125263)
            + 1.02539759976548;
    }
    if engine == "LiblineaR" {
        log_fold += (lo * -0.343818313721125) + (hi * -5.25579902213519) + -0.204133402827292;
    }
    if engine == "lightgbm" {
        log_fold += hi * -7.14043416548319;
    }
    log_fold > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xgboost_decision_follows_workflows() {
        // ames_reg_xgb in the reference run: estimate 0.836, 2197 rows -> "yes"
        assert!(should_use_sparsity(0.8360204, "xgboost", 2197));
        // a dense synthetic design -> "no"
        assert!(!should_use_sparsity(0.26, "xgboost", 30));
        assert!(!should_use_sparsity(0.99, "ranger", 2197));
    }
}

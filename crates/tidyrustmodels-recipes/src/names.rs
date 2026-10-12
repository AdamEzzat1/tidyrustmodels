//! R-compatible name sanitising (`make.names`) and dummy-column naming.

/// Reserved words that R's `make.names` suffixes with a dot.
const R_RESERVED: &[&str] = &[
    "if",
    "else",
    "repeat",
    "while",
    "function",
    "for",
    "next",
    "break",
    "TRUE",
    "FALSE",
    "NULL",
    "Inf",
    "NaN",
    "NA",
    "NA_integer_",
    "NA_real_",
    "NA_character_",
    "in",
];

/// Port of R's `make.names(x, unique = FALSE)` for ASCII input: invalid characters
/// become `.`, a leading digit/underscore/dot-digit gets an `X` prefix, and reserved
/// words get a trailing `.`.
///
/// Non-ASCII letters are treated as invalid characters (R's behaviour depends on the
/// locale; the C locale treats them as invalid). Fixture tests cover the levels that
/// occur in the reference datasets.
pub fn make_names(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                c
            } else {
                '.'
            }
        })
        .collect();
    let needs_prefix = match out.chars().next() {
        None => true,
        Some(c) if c.is_ascii_digit() || c == '_' => true,
        Some('.') => out.chars().nth(1).is_some_and(|d| d.is_ascii_digit()),
        Some(_) => false,
    };
    if needs_prefix {
        out.insert(0, 'X');
    }
    if R_RESERVED.contains(&out.as_str()) {
        out.push('.');
    }
    out
}

/// recipes' default `dummy_names(var, lvl)`: `paste(var, make.names(lvl), sep = "_")`.
pub fn dummy_name(var: &str, level: &str) -> String {
    format!("{var}_{}", make_names(level))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_make_names_cases() {
        assert_eq!(make_names("setosa"), "setosa");
        assert_eq!(make_names("One Story"), "One.Story");
        assert_eq!(make_names("1Fam"), "X1Fam");
        assert_eq!(make_names("_x"), "X_x");
        assert_eq!(make_names(".5"), "X.5");
        assert_eq!(make_names(".x"), ".x");
        assert_eq!(make_names("if"), "if.");
        assert_eq!(make_names(""), "X");
        assert_eq!(make_names("a-b+c"), "a.b.c");
    }

    #[test]
    fn dummy_names_match_recipes() {
        assert_eq!(dummy_name("species", "Chinstrap"), "species_Chinstrap");
        assert_eq!(dummy_name("Bldg_Type", "1Fam"), "Bldg_Type_X1Fam");
    }
}

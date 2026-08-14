//! Elaboration of [`crate::ast::region`] (Language Reference chapter 12): ludemes returning a
//! region (a collection of sites).
//!
//! Only `(sites Empty)` (12.4.4, the `sitesIndexType::Empty` form of the `(sites ...)` "super
//! ludeme", with no `[<siteType>]`/`[<int>]` qualifiers) is elaborated so far, since it's all
//! [`crate::elaborate::common`] needs for `(to (sites Empty))`.

use crate::ast::located::Located;
use crate::ast::region::{RegionFunction, Sites, SitesIndexType};
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::SExpr;

/// `(sites Empty)` (12.4.4).
pub fn elaborate_sites(v: &Located<SExpr>) -> Result<Sites, ElaborateError> {
    let call = call_ident(v, "sites")?;
    let arg = one_positional_arg(call, v.span)?;
    let SExpr::Ident(kind) = &arg.node else {
        return Err(ElaborateError {
            message: format!("expected a sitesIndexType identifier, found {:?}", arg.node),
            span: arg.span,
        });
    };
    match kind.as_str() {
        "Empty" => Ok(Sites::Index {
            kind: SitesIndexType::Empty,
            site_type: None,
            index: None,
        }),
        other => Err(ElaborateError {
            message: format!("unsupported (sites {other} ...) -- only Empty is elaborated so far"),
            span: arg.span,
        }),
    }
}

/// Any ludeme producing a region. Only [`elaborate_sites`] is wired up so far.
pub fn elaborate_region_function(v: &Located<SExpr>) -> Result<RegionFunction, ElaborateError> {
    Ok(RegionFunction::Sites(Box::new(elaborate_sites(v)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn sites_empty() {
        assert_eq!(
            elaborate_sites(&parse_one("(sites Empty)")).unwrap(),
            Sites::Index {
                kind: SitesIndexType::Empty,
                site_type: None,
                index: None,
            }
        );
    }

    #[test]
    fn region_function_wraps_sites() {
        assert!(matches!(
            elaborate_region_function(&parse_one("(sites Empty)")).unwrap(),
            RegionFunction::Sites(_)
        ));
    }

    #[test]
    fn unsupported_sites_kind_errors() {
        assert!(elaborate_sites(&parse_one("(sites Occupied by:Mover)")).is_err());
    }
}

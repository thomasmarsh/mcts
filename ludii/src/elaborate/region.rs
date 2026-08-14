//! Elaboration of [`crate::ast::region`] (Language Reference chapter 12): ludemes returning a
//! region (a collection of sites).
//!
//! Only `(sites Empty)` (12.4.4, the `sitesIndexType::Empty` form of the `(sites ...)` "super
//! ludeme", with no `[<siteType>]`/`[<int>]` qualifiers) and `(sites Side <compassDirection>)`
//! (12.4.2's `Side` form, with no `[<siteType>]`) are elaborated so far, since between them
//! that's all [`crate::elaborate::common`]/[`crate::elaborate::equipment`] need for `(to (sites
//! Empty))` and Hex's `(regions P1 {(sites Side NE) (sites Side SW)})`.

use crate::ast::located::Located;
use crate::ast::region::{RegionFunction, SideTarget, Sites, SitesIndexType};
use crate::elaborate::types::elaborate_compass_direction;
use crate::elaborate::{call_ident, ElaborateError};
use crate::parse::SExpr;

pub fn elaborate_sites(v: &Located<SExpr>) -> Result<Sites, ElaborateError> {
    let call = call_ident(v, "sites")?;
    let mut positional = call.args.iter().filter(|a| a.name.is_none());
    let kind_arg = positional.next().ok_or_else(|| ElaborateError {
        message: "(sites ...) requires a sitesIndexType/kind argument".into(),
        span: v.span,
    })?;
    let SExpr::Ident(kind) = &kind_arg.value.node else {
        return Err(ElaborateError {
            message: format!(
                "expected a sitesIndexType identifier, found {:?}",
                kind_arg.value.node
            ),
            span: kind_arg.value.span,
        });
    };
    match kind.as_str() {
        "Empty" => Ok(Sites::Index {
            kind: SitesIndexType::Empty,
            site_type: None,
            index: None,
        }),
        "Side" => {
            let compass_arg = positional.next().ok_or_else(|| ElaborateError {
                message: "(sites Side ...) requires a compassDirection argument".into(),
                span: v.span,
            })?;
            let compass = elaborate_compass_direction(&compass_arg.value)?;
            Ok(Sites::Side {
                site_type: None,
                target: Some(SideTarget::Compass(compass)),
            })
        }
        other => Err(ElaborateError {
            message: format!(
                "unsupported (sites {other} ...) -- only Empty and Side are elaborated so far"
            ),
            span: kind_arg.value.span,
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
    fn sites_side() {
        assert_eq!(
            elaborate_sites(&parse_one("(sites Side NE)")).unwrap(),
            Sites::Side {
                site_type: None,
                target: Some(SideTarget::Compass(crate::ast::types::CompassDirection::NE)),
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

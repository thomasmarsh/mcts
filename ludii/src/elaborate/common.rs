//! Elaboration of the small utility ludemes in [`crate::ast::common`] that are threaded through
//! almost every move ludeme.
//!
//! Only `(to <region>)` (15.6.6, region-valued location only -- no `[<siteType>]`, `level:`,
//! `[<rotations>]`, `if:`, or `[<apply>]`) is elaborated so far, since it's all
//! [`crate::elaborate::moves::decision`] needs for `(to (sites Empty))`.

use crate::ast::common::{SiteOrRegion, To};
use crate::ast::located::Located;
use crate::elaborate::region::elaborate_region_function;
use crate::elaborate::{call_ident, one_positional_arg, ElaborateError};
use crate::parse::SExpr;

/// `(to <region>)` (15.6.6), region-valued location only.
pub fn elaborate_to(v: &Located<SExpr>) -> Result<To, ElaborateError> {
    let call = call_ident(v, "to")?;
    let arg = one_positional_arg(call, v.span)?;
    let region = elaborate_region_function(arg)?;
    Ok(To {
        site_type: None,
        location: Some(SiteOrRegion::Region(Box::new(Located::new(
            region, arg.span,
        )))),
        level: None,
        rotations: None,
        condition: None,
        apply: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::region::{RegionFunction, Sites, SitesIndexType};
    use crate::parse::parse;

    fn parse_one(src: &str) -> Located<SExpr> {
        let mut forms = parse(src).unwrap();
        assert_eq!(forms.len(), 1);
        forms.remove(0)
    }

    #[test]
    fn to_sites_empty() {
        let to = elaborate_to(&parse_one("(to (sites Empty))")).unwrap();
        let Some(SiteOrRegion::Region(region)) = &to.location else {
            panic!("expected a region location");
        };
        let RegionFunction::Sites(sites) = &region.node else {
            panic!("expected Sites");
        };
        assert_eq!(
            **sites,
            Sites::Index {
                kind: SitesIndexType::Empty,
                site_type: None,
                index: None,
            }
        );
    }
}

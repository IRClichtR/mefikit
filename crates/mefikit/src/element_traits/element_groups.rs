//! Group membership queries for mesh elements.
//!
//! Provides the [`ElementGroups`] trait for zero-copy, on-demand group membership
//! queries. Group membership is computed from the element's family ID and the
//! block's groups map — no data is cached or allocated. The groups map is held
//! privately (like the coords table) and never exposed through the trait.

/// Zero-copy group membership queries for mesh elements.
///
/// Group membership is computed on demand from the element's family ID and the
/// block's groups map, following the same pattern as coordinate access
/// (`element.coord(i)`). The underlying groups map is held privately and never
/// exposed — only computed queries are available.
pub trait ElementGroups<'a> {
    /// Returns the family ID of this element.
    fn family_id(&self) -> usize;

    /// Returns true if this element belongs to the named group.
    ///
    /// Computed on demand: looks up whether the element's family ID is in the
    /// group's family ID set. O(log g) where g is the number of groups.
    /// No allocation.
    fn in_group(&self, group: &str) -> bool;

    /// Returns an iterator over the names of all groups this element belongs to.
    ///
    /// Computed on demand: scans the groups map yielding references to group
    /// names whose family ID set contains this element's family ID. Zero-copy
    /// (yields `&str` references).
    fn groups(&self) -> impl Iterator<Item = &'a str>;
}

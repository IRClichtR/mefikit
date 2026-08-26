//! Topological operations for mesh elements.
//!
//! Provides the [`ElementTopo`] trait for extracting subentities (faces, edges, vertices)
//! and decomposing elements into simplexes.

use ndarray::prelude::*;

use crate::mesh::Connectivity;
use crate::mesh::{Dimension, ElementLike, ElementType};

/// Topological operations for mesh elements.
///
/// Extends [`ElementLike`] with methods for extracting subentities at various
/// codimensions and decomposing elements into simplex components.
pub trait ElementTopo<'a>: ElementLike<'a> {
    /// Returns the subentities of the element at the given codimension.
    ///
    /// For example, for a QUAD4 element:
    /// - `codim = D1` returns the 4 edges (SEG2)
    /// - `codim = D2` returns the 4 vertices (VERTEX)
    ///
    /// If `codim` is `None`, defaults to `D1`.
    fn subentities(&self, codim: Option<Dimension>) -> Vec<(ElementType, Connectivity)> {
        use ElementType::*;
        let codim = match codim {
            None => Dimension::D1,
            Some(c) => c,
        };
        let co = self.connectivity();
        let mut res = Vec::new();
        match self.element_type() {
            SEG2 | SEG3 | SEG4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[[co[0]], [co[1]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim different from D1 on SEG"),
            },
            TRI3 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[[co[0], co[1]], [co[1], co[2]], [co[2], co[0]]]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on TRI3"),
            },
            TRI6 | TRI7 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[3]],
                        [co[1], co[2], co[4]],
                        [co[2], co[0], co[5]],
                    ]);
                    res.push((SEG3, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on TRI3"),
            },
            QUAD4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[1], co[2]],
                        [co[2], co[3]],
                        [co[3], co[0]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]], [co[3]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on QUAD"),
            },
            TET4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[2]],
                        [co[1], co[2], co[3]],
                        [co[2], co[3], co[0]],
                        [co[3], co[0], co[1]],
                    ]);
                    res.push((TRI3, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[0], co[2]],
                        [co[0], co[3]],
                        [co[1], co[2]],
                        [co[1], co[3]],
                        [co[2], co[3]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D3 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]], [co[3]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => {
                    panic!("It is not possible to ask for codim diff from D1, D2 or D3 on TET")
                }
            },
            HEX8 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[2], co[3]],
                        [co[0], co[3], co[7], co[4]],
                        [co[0], co[4], co[5], co[1]],
                        [co[1], co[5], co[6], co[2]],
                        [co[2], co[6], co[7], co[3]],
                        [co[4], co[7], co[6], co[5]],
                    ]);
                    res.push((QUAD4, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[0], co[3]],
                        [co[0], co[4]],
                        [co[1], co[2]],
                        [co[1], co[5]],
                        [co[2], co[3]],
                        [co[2], co[6]],
                        [co[3], co[7]],
                        [co[4], co[5]],
                        [co[4], co[7]],
                        [co[5], co[6]],
                        [co[6], co[7]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D3 => {
                    let conn = arr2(&[
                        [co[0]],
                        [co[1]],
                        [co[2]],
                        [co[3]],
                        [co[4]],
                        [co[5]],
                        [co[6]],
                        [co[7]],
                    ]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => {
                    panic!("It is not possible to ask for codim diff from D1, D2 or D3 on HEX")
                }
            },
            PGON => match codim {
                Dimension::D1 => {
                    let mut conn: Vec<_> = co.windows(2).flatten().cloned().collect();
                    conn.push(co[co.len() - 1]);
                    conn.push(co[0]);
                    let conn = Array2::from_shape_vec([conn.len() / 2, 2], conn).unwrap();
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = Array2::from_shape_vec([co.len(), 1], co.to_vec()).unwrap();
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 or D2 on PGON"),
            },
            PHED => match codim {
                Dimension::D1 => {
                    let mut conn = Vec::new();
                    let mut offsets = Vec::new();
                    let mut offset = 0;
                    co.split_inclusive(|&e| e == usize::MAX).for_each(|a| {
                        let len = a.len() - 1;
                        offset += len;
                        offsets.push(offset);
                        conn.append(&mut a[..len].to_vec())
                    });
                    let offsets = Array1::from_vec(offsets);
                    let conn = Array::from_vec(conn);
                    res.push((
                        PGON,
                        Connectivity::new_poly(conn.to_shared(), offsets.to_shared()),
                    ));
                }
                _ => {
                    todo!()
                }
            },
            _ => todo!(), // For other types, return empty vector
        };
        res
    }

    /// Converts the element to its polygonal/polyhedral equivalent.
    ///
    /// - 0D (VERTEX) is returned unchanged.
    /// - 1D elements become SPLINE with the same node list.
    /// - 2D elements become PGON with the same node list.
    /// - 3D elements become PHED with face-based connectivity. Faces are
    ///   separated by `usize::MAX` sentinel values in the returned vector.
    /// - Already-poly elements are returned unchanged.
    fn to_poly(&self) -> (ElementType, Vec<usize>) {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX => (VERTEX, co.to_vec()),
            SEG2 | SEG3 | SEG4 => (SPLINE, co.to_vec()),
            TRI3 | TRI6 | TRI7 | QUAD4 | QUAD8 | QUAD9 => (PGON, co.to_vec()),
            TET4 => {
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[2], m, co[1], co[2], co[3], m, co[2], co[3], co[0], m,
                        co[3], co[0], co[1],
                    ],
                )
            }
            TET10 => {
                // VTK TET10: 0-3 vertices, 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[2], co[4], co[5], co[6], m, co[1], co[2], co[3], co[5],
                        co[9], co[8], m, co[2], co[3], co[0], co[9], co[7], co[6], m, co[3], co[0],
                        co[1], co[7], co[4], co[8],
                    ],
                )
            }
            HEX8 => {
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[2], co[3], m, co[0], co[3], co[7], co[4], m, co[0], co[4],
                        co[5], co[1], m, co[1], co[5], co[6], co[2], m, co[2], co[6], co[7], co[3],
                        m, co[4], co[7], co[6], co[5],
                    ],
                )
            }
            HEX21 => {
                // VTK HEX21: 0-7 vertices, 8(0-1), 9(1-2), 10(2-3), 11(3-0),
                //   12(4-5), 13(5-6), 14(6-7), 15(7-4), 16(0-4), 17(1-5), 18(2-6), 19(3-7)
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[2], co[3], co[8], co[9], co[10], co[11], m, co[0], co[3],
                        co[7], co[4], co[11], co[19], co[15], co[16], m, co[0], co[4], co[5],
                        co[1], co[16], co[12], co[17], co[8], m, co[1], co[5], co[6], co[2],
                        co[17], co[13], co[18], co[9], m, co[2], co[6], co[7], co[3], co[18],
                        co[14], co[19], co[10], m, co[4], co[7], co[6], co[5], co[15], co[14],
                        co[13], co[12],
                    ],
                )
            }
            SPLINE | PGON | PHED => (self.element_type(), co.to_vec()),
        }
    }

    /// Decomposes the element into simplex elements.
    ///
    /// Returns a list of (element type, connectivity) tuples representing
    /// the simplex decomposition. For example, a QUAD4 is decomposed into
    /// two TRI3 elements.
    fn to_simplexes(&self) -> Vec<(ElementType, Vec<usize>)> {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX => vec![(VERTEX, vec![co[0]])],
            SEG2 | SEG3 | SEG4 => vec![(SEG2, vec![co[0], co[1]])],
            TRI3 | TRI6 | TRI7 => vec![(TRI3, vec![co[0], co[1], co[2]])],
            QUAD4 | QUAD8 | QUAD9 => vec![
                (TRI3, vec![co[0], co[1], co[3]]),
                (TRI3, vec![co[2], co[3], co[1]]),
            ],
            TET4 | TET10 => vec![(TET4, vec![co[0], co[1], co[2], co[3]])],
            HEX8 | HEX21 => vec![
                (TET4, vec![co[0], co[1], co[3], co[4]]),
                (TET4, vec![co[2], co[3], co[1], co[6]]),
                (TET4, vec![co[7], co[6], co[4], co[3]]),
                (TET4, vec![co[5], co[4], co[6], co[1]]),
                (TET4, vec![co[4], co[6], co[3], co[1]]),
            ],
            _ => todo!(),
        }
    }
}

impl<'a, T> ElementTopo<'a> for T where T: ElementLike<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Element, ElementType};
    use ndarray as nd;

    #[test]
    fn test_subentities_quad4_codim1() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D1));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 4 edges
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::SEG2);
        // Check that connectivity contains 4 edges (4 x 2 nodes = 8 values)
        assert_eq!(connectivity.len(), 4);
    }

    #[test]
    fn test_subentities_quad4_codim2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D2));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 4 vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 4);
    }

    #[test]
    fn test_subentities_tri3_codim1() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D1));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 3 edges
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::SEG2);
        assert_eq!(connectivity.len(), 3);
    }

    #[test]
    fn test_subentities_tri3_codim2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D2));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 3 vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 3);
    }

    #[test]
    fn test_subentities_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SEG2,
            &groups,
        );
        let subentities = elem.subentities(None); // defaults to D1
        assert_eq!(subentities.len(), 1); // One Connectivity containing both vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 2);
    }

    #[test]
    fn test_to_simplexes_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        let simplexes = elem.to_simplexes();
        assert_eq!(simplexes.len(), 2); // QUAD4 -> 2 TRI3
        for (et, _) in &simplexes {
            assert_eq!(*et, ElementType::TRI3);
        }
    }

    #[test]
    fn test_to_simplexes_tri3() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let simplexes = elem.to_simplexes();
        assert_eq!(simplexes.len(), 1); // TRI3 -> 1 TRI3
        assert_eq!(simplexes[0].0, ElementType::TRI3);
    }

    #[test]
    fn test_to_poly_vertex() {
        let coords = nd::array![[0.0, 0.0]];
        let conn = &[0];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::VERTEX,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::VERTEX);
        assert_eq!(poly_conn, vec![0]);
    }

    #[test]
    fn test_to_poly_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SEG2,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::SPLINE);
        assert_eq!(poly_conn, vec![0, 1]);
    }

    #[test]
    fn test_to_poly_tri3() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }

    #[test]
    fn test_to_poly_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_to_poly_tet4() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TET4,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);
        // 4 faces x 3 nodes + 3 separators = 15 entries
        assert_eq!(poly_conn.len(), 15);
        // Check face separators
        assert_eq!(poly_conn[3], usize::MAX);
        assert_eq!(poly_conn[7], usize::MAX);
        assert_eq!(poly_conn[11], usize::MAX);
        // Check first face
        assert_eq!(&poly_conn[0..3], &[0, 1, 2]);
    }

    #[test]
    fn test_to_poly_hex8() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3, 4, 5, 6, 7];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::HEX8,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);
        // 6 faces x 4 nodes + 5 separators = 29 entries
        assert_eq!(poly_conn.len(), 29);
        // Check first face
        assert_eq!(&poly_conn[0..4], &[0, 1, 2, 3]);
        // Check last face
        assert_eq!(&poly_conn[25..29], &[4, 7, 6, 5]);
    }

    #[test]
    fn test_to_poly_pgon_unchanged() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::PGON,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }

    #[test]
    fn test_to_poly_spline_unchanged() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SPLINE,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::SPLINE);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }
}

use std::path::Path;
use hdf5_metno::{File, Group, Dataset};
use hdf5_metno::types::{FixedAscii, };
use ndarray::{Array1, Array2, arr1, array};
use std::collections::BTreeMap;
use crate::mesh::{ElementType, UMesh, UMeshView};

fn read_string_data(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let s: String = group
        .dataset(" data")?
        .as_reader()
        .read_1d::<i8>()?
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8 as char)
        .collect();
    Ok(s.trim().to_string())
}

fn cgns_label(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr(" label").or_else(|_| group.attr("label"))?;
    let label: String = attr
        .as_reader()
        .read_scalar::<FixedAscii<64>>()?
        .to_string();
    Ok(label.trim().trim_matches('\0').to_string())
}

fn find_first_child_with_label(
    group: &Group,
    label: &str,
) -> Result<Group, Box<dyn std::error::Error>> {
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        if lbl == label {
            return Ok(child);
        }
    }
    Err(format!("no child with label '{label}' in '{}'", group.name()).into())
}

fn children_with_label(
    group: &Group,
    label: &str,
) -> Result<Vec<Group>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        if lbl == label {
            out.push(child);
        }
    }
    Ok(out)
}

fn cgns_code_to_element_type(code: i32) -> Option<ElementType> {
    match code {
        2  => Some(ElementType::VERTEX),
        3  => Some(ElementType::SEG2),
        5  => Some(ElementType::TRI3),
        7  => Some(ElementType::QUAD4),
        10 => Some(ElementType::TET4),
        17 => Some(ElementType::HEX8),
        22 => Some(ElementType::PGON),
        23 => Some(ElementType::PHED),
        other => {
            eprintln!("warning: unsupported CGNS element type {other}, section skipped");
            None
        }
    }
}

fn element_type_to_cgns(et: ElementType) -> i32 {
    match et {
        ElementType::VERTEX => 2,
        ElementType::SEG2   => 3,
        ElementType::TRI3   => 5,
        ElementType::QUAD4  => 7,
        ElementType::TET4   => 10,
        ElementType::HEX8   => 17,
        ElementType::PGON   => 22,
        ElementType::PHED   => 23,
        other => panic!("unsupported ElementType {other:?}"),
    }
}

fn nodes_per_cgns_code(code: i32) -> Option<usize> {
    match code {
        2  => Some(1),
        3  => Some(2),
        5  => Some(3),
        7  => Some(4),
        10 => Some(4),
        17 => Some(8),
        _  => None, // poly types have variable stride — handled separately
    }
}

fn read_coordinates(
    zone: &Group,
    phys_dim: usize,
) -> Result<ndarray::ArcArray2<f64>, Box<dyn std::error::Error>> {
    let gc = find_first_child_with_label(zone, "GridCoordinates_t")?;
    let names = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    let columns: Vec<Vec<f64>> = (0..phys_dim)
        .map(|i| {
            // CoordinateX/Y/Z are groups, data lives in their " data" dataset
            let coord_group = gc.group(names[i])
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            let ds = coord_group.dataset(" data")
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

            // handle both R4 and R8 — check "type" attribute
            let type_attr = coord_group
                .attr("type")
                .and_then(|a| {
                    use hdf5_metno::types::FixedAscii;
                    a.as_reader().read_scalar::<FixedAscii<8>>()
                })
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "R8".to_string());

            let values: Vec<f64> = if type_attr.trim_matches('\0').starts_with("R4") {
                ds.as_reader()
                    .read_1d::<f32>()?
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            } else {
                ds.as_reader()
                    .read_1d::<f64>()?
                    .to_vec()
            };

            Ok(values)
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    let n = columns[0].len();
    let mut coords = ndarray::Array2::<f64>::zeros((n, phys_dim));
    for (col, arr) in columns.iter().enumerate() {
        for i in 0..n {
            coords[[i, col]] = arr[i];
        }
    }
    Ok(coords.into_shared())
}

// first pass: collect BC ranges/pointlists → map global_cgns_idx → family_id
// before reading elements
fn collect_bc_families(
    zone: &Group,
) -> Result<std::collections::HashMap<i32, usize>, Box<dyn std::error::Error>> {
    let mut map = std::collections::HashMap::new();

    let Ok(zonebc) = find_first_child_with_label(zone, "ZoneBC_t") else {
        return Ok(map);
    };

    for (family_id, bc) in children_with_label(&zonebc, "BC_t")?
        .into_iter()
        .enumerate()
        .map(|(i, bc)| (i + 1, bc))
    {
        let face_ids: Vec<i32> = if let Ok(pl) = bc.group("PointList") {
            pl.dataset(" data")?
                .as_reader()
                .read_dyn::<i32>()?
                .into_raw_vec_and_offset().0
        } else if let Ok(pr) = bc.group("PointRange") {
            let flat = pr.dataset(" data")?
                .as_reader()
                .read_dyn::<i32>()?
                .into_raw_vec_and_offset().0;
            (flat[0]..=flat[1]).collect()
        } else {
            continue;
        };

        for idx in face_ids {
            map.insert(idx, family_id);
        }
    }

    Ok(map)
}

// Read element uses bc_families collected
fn read_elements(
    zone: &Group,
    mesh: &mut UMesh,
    bc_families: &std::collections::HashMap<i32, usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut sections = children_with_label(zone, "Elements_t")?;

    sections.sort_by_key(|s| {
        find_first_child_with_label(s, "IndexRange_t")
            .and_then(|r| r.dataset(" data")
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>))
            .and_then(|d| d.as_reader().read_dyn::<i32>()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>))
            .map(|a| a.into_raw_vec_and_offset().0[0])
            .unwrap_or(i32::MAX)
    });

    let mut global_idx = 1_i32; // 1-based running counter

    for section in &sections {
        let meta: Vec<i32> = section
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i32>()?
            .into_raw_vec_and_offset().0;
        let cgns_code = meta[0];

        let Some(elem_type) = cgns_code_to_element_type(cgns_code) else {
            // count skipped elements to keep global_idx accurate
            let range: Vec<i32> = find_first_child_with_label(section, "IndexRange_t")?
                .dataset(" data")?
                .as_reader()
                .read_dyn::<i32>()?
                .into_raw_vec_and_offset().0;
            global_idx += range[1] - range[0] + 1;
            eprintln!("warning: skipping unsupported CGNS type {cgns_code}");
            continue;
        };

        let conn: Vec<i32> = section
            .group("ElementConnectivity")?
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i32>()?
            .into_raw_vec_and_offset().0;

        match nodes_per_cgns_code(cgns_code) {
            Some(stride) => {
                for chunk in conn.chunks(stride) {
                    let family = bc_families.get(&global_idx).copied();
                    let nodes: Vec<usize> = chunk.iter().map(|&n| (n - 1) as usize).collect();
                    mesh.add_element(elem_type, &nodes, family, None);
                    global_idx += 1;
                }
            }
            None => {
                // length-prefixed poly (NGON_n, NFACE_n)
                let mut i = 0;
                while i < conn.len() {
                    let n_nodes = conn[i] as usize;
                    i += 1;
                    if i + n_nodes > conn.len() {
                        eprintln!("warning: malformed poly connectivity, stopping");
                        break;
                    }
                    let family = bc_families.get(&global_idx).copied();
                    let nodes: Vec<usize> = conn[i..i + n_nodes]
                        .iter()
                        .map(|&v| (v - 1) as usize)
                        .collect();
                    mesh.add_element(elem_type, &nodes, family, None);
                    i += n_nodes;
                    global_idx += 1;
                }
            }
        }
    }

    Ok(())
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;

    // base
    dbg!("Read base");
    let base = find_first_child_with_label(&file.as_group()?, "CGNSBase_t")?;
    let base_data: Vec<i32> = base
        .dataset(" data")?
        .as_reader()
        .read_dyn::<i32>()?
        .into_raw_vec_and_offset().0;
    let phys_dim = base_data[1] as usize;

    // zone
    dbg!("Read zone");
    let zone = find_first_child_with_label(&base, "Zone_t")?;

    // zone type check
    dbg!("zone type check");
    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(format!("unsupported zone type: {z_type}").into());
    }

    // coordinates
    dbg!("read coords");
    let coords = read_coordinates(&zone, phys_dim)?;
    let mut mesh = UMesh::new(coords);

    // collect BC family assignments before adding elements
    dbg!("collect BC families");
    let bc_families = collect_bc_families(&zone)?;

    // add elements with family tags already resolved
    dbg!("read elements with family tags");
    read_elements(&zone, &mut mesh, &bc_families)?;

    Ok(mesh)
}
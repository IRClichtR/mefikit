use crate::mesh::ConnectivityView;
use crate::mesh::ElementBlock;
use crate::mesh::ElementBlockView;
use crate::mesh::ElementType;
use crate::mesh::IntoElementBlockEntry;
use crate::mesh::Regularity;
use crate::mesh::UMesh;
use crate::mesh::UMeshView;

use hdf5_metno::types::FixedAscii;
use hdf5_metno::types::VarLenAscii;
use hdf5_metno::{File, Group};
use ndarray::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

impl ElementType {
    pub fn med_name(self) -> &'static str {
        match self {
            ElementType::VERTEX => "PO1",
            ElementType::SEG2 => "SE2",
            ElementType::SEG3 => "SE3",

            ElementType::TRI3 => "TR3",
            ElementType::TRI6 => "TR6",
            ElementType::TRI7 => "TR7",

            ElementType::QUAD4 => "QU4",
            ElementType::QUAD8 => "QU8",
            ElementType::QUAD9 => "QU9",

            ElementType::TET4 => "TE4",
            ElementType::TET10 => "T10",

            ElementType::HEX8 => "HE8",
            // ElementType::HEX21     => "H20",
            // ElementType::Hexa27     => "H27",

            // ElementType::Pyramid5   => "PY5",
            // ElementType::Pyramid13  => "P13",

            // ElementType::Wedge6     => "PE6",
            // ElementType::Wedge15    => "P15",
            ElementType::PGON => "POG",
            _ => todo!(),
        }
    }
}

pub fn write(
    path: impl AsRef<std::path::Path>,
    mesh: &UMeshView,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = hdf5_metno::File::create(path)?;

    file.create_group("CHA")?;
    write_general_info(&file)?;
    write_mesh(&file, mesh)?;

    Ok(())
}

fn write_general_info(file: &File) -> hdf5_metno::Result<()> {
    let info = file.create_group("INFOS_GENERALES")?;

    write_scalar_attr(&info, "MAJ", 3i64)?;
    write_scalar_attr(&info, "MIN", 0i64)?;
    write_scalar_attr(&info, "REL", 0i64)?;

    Ok(())
}

fn write_scalar_attr<T: hdf5_metno::H5Type>(
    obj: &hdf5_metno::Location,
    name: &str,
    value: T,
) -> hdf5_metno::Result<()> {
    obj.new_attr::<T>().create(name)?.write_scalar(&value)
}

fn write_mesh(file: &File, mesh: &UMeshView) -> hdf5_metno::Result<()> {
    let ensemble = file.create_group("ENS_MAA")?;
    let med_mesh = ensemble.create_group("mesh")?;

    let dim = mesh.coords().shape()[1];

    write_scalar_attr(&med_mesh, "DIM", dim as i64)?;
    write_scalar_attr(&med_mesh, "ESP", dim as i64)?;
    write_scalar_attr(&med_mesh, "REP", 0i64)?;

    // Empty for now, as in the Python reference when no units are supplied.
    write_fixed_bytes_attr::<1>(&med_mesh, "UNT", b"")?;
    write_fixed_bytes_attr::<1>(&med_mesh, "UNI", b"")?;

    write_scalar_attr(&med_mesh, "SRT", 1i64)?;

    // "X", "Y", "Z", each occupying 16 characters.
    let names = ["X", "Y", "Z"];
    let mut nom = Vec::with_capacity(dim * 16);

    for name in names.iter().take(dim) {
        let bytes = name.as_bytes();
        nom.extend_from_slice(bytes);
        nom.resize(nom.len() + 16 - bytes.len(), b' ');
    }

    write_fixed_bytes_attr::<48>(&med_mesh, "NOM", &nom)?;

    write_fixed_bytes_attr::<25>(&med_mesh, "DES", b"Mesh created with mefikit")?;

    write_scalar_attr(&med_mesh, "TYP", 0i64)?;

    let timestep = med_mesh.create_group("-0000000000000000001-0000000000000000001")?;

    write_scalar_attr(&timestep, "CGT", 1i64)?;
    write_scalar_attr(&timestep, "NDT", -1i64)?;
    write_scalar_attr(&timestep, "NOR", -1i64)?;
    write_scalar_attr(&timestep, "PDT", -1.0f64)?;

    // Explicitly create CHA, even though it is empty for now.

    write_nodes(&timestep, &mesh.coords())?;

    let mai = timestep.create_group("MAI")?;
    write_scalar_attr(&mai, "CGT", 1i64)?;

    for (et, block) in mesh.blocks() {
        write_block(&mai, et, &block.view())?;
    }

    write_families(file)?;

    Ok(())
}

fn write_nodes(timestep: &Group, coords: &ArrayView2<f64>) -> hdf5_metno::Result<()> {
    let noe = timestep.create_group("NOE")?;

    write_scalar_attr(&noe, "CGT", 1i64)?;
    write_scalar_attr(&noe, "CGS", 1i64)?;

    write_fixed_bytes_attr::<23>(&noe, "PFL", b"MED_NO_PROFILE_INTERNAL")?;

    let coo_data: Vec<f64> = {
        let (n, dim) = coords.dim();
        let mut out = Vec::with_capacity(n * dim);

        for d in 0..dim {
            for n in 0..n {
                out.push(coords[[n, d]]);
            }
        }

        out
    };

    let coo = noe
        .new_dataset_builder()
        .with_data(&coo_data)
        .create("COO")?;

    write_scalar_attr(&coo, "CGT", 1i64)?;
    write_scalar_attr(&coo, "NBR", coords.nrows() as i64)?;

    Ok(())
}

impl ElementType {
    fn med_permutation(self) -> Option<&'static [usize]> {
        match self {
            ElementType::TET4 => Some(&[0, 1, 3, 2]),

            // ElementType::HEX8 => Some(&[4, 5, 6, 7, 0, 1, 2, 3]),
            ElementType::HEX8 => Some(&[0, 1, 2, 3, 4, 5, 6, 7]),

            ElementType::TET10 => Some(&[0, 1, 3, 2, 4, 8, 7, 6, 5, 9]),

            _ => None,
        }
    }
}

fn reorder_connectivity(conn: &ArrayView2<usize>, permutation: Option<&[usize]>) -> Array2<u64> {
    let Some(p) = permutation else {
        return conn.mapv(|x| x as u64);
    };

    let mut out = Array2::<u64>::zeros((conn.nrows(), p.len()));

    for (new_j, &old_j) in p.iter().enumerate() {
        out.column_mut(new_j)
            .assign(&conn.column(old_j).mapv(|x| x as u64));
    }

    out
}

fn write_regular(
    mai: &Group,
    element_type: ElementType,
    regular: &ElementBlockView,
) -> hdf5_metno::Result<()> {
    let conn = match &regular.connectivity {
        ConnectivityView::Regular(pc) => pc,
        _ => panic!("A Regular block must contain a regular connectivity."),
    };
    let med_type = element_type.med_name();

    let group = mai.create_group(med_type)?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;
    write_fixed_bytes_attr::<23>(&group, "PFL", b"MED_NO_PROFILE_INTERNAL")?;
    let med_conn = reorder_connectivity(conn, element_type.med_permutation());

    let mut flat = Vec::with_capacity(med_conn.len());

    // MED expects column-major ordering.
    for j in 0..med_conn.ncols() {
        for i in 0..med_conn.nrows() {
            flat.push(med_conn[[i, j]] + 1);
        }
    }

    let nod = group.new_dataset_builder().with_data(&flat).create("NOD")?;

    write_scalar_attr(&nod, "CGT", 1i64)?;
    write_scalar_attr(&nod, "NBR", med_conn.nrows() as i64)?;

    Ok(())
}

fn write_families(file: &File) -> hdf5_metno::Result<()> {
    let fas = file.create_group("FAS")?;
    let families = fas.create_group("mesh")?;

    let family_zero = families.create_group("FAMILLE_ZERO")?;
    write_scalar_attr(&family_zero, "NUM", 0i64)?;

    Ok(())
}

#[allow(unused)]
fn write_bytes_attr(group: &Group, name: &str, value: &[u8]) -> hdf5_metno::Result<()> {
    let value =
        VarLenAscii::from_ascii(value).map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    group
        .new_attr::<VarLenAscii>()
        .create(name)?
        .write_scalar(&value)?;

    Ok(())
}

fn write_fixed_bytes_attr<const N: usize>(
    group: &Group,
    name: &str,
    value: &[u8],
) -> hdf5_metno::Result<()> {
    let value = FixedAscii::<N>::from_ascii(value)
        .map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    group
        .new_attr::<FixedAscii<N>>()
        .create(name)?
        .write_scalar(&value)
}

fn write_polygon(mai: &Group, poly: &ElementBlockView) -> hdf5_metno::Result<()> {
    let poly_conn = match &poly.connectivity {
        ConnectivityView::Poly(pc) => pc,
        _ => panic!("A PGON block must contain a poly connectivity."),
    };
    let group = mai.create_group("POG")?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;

    // MED NOD is 1-based.
    let nod: Vec<u64> = poly_conn.data.iter().map(|&x| x as u64 + 1).collect();

    let nod_ds = group.new_dataset_builder().with_data(&nod).create("NOD")?;

    write_scalar_attr(&nod_ds, "CGT", 1i64)?;
    write_scalar_attr(&nod_ds, "NBR", nod.len() as i64)?;

    // Mefikit:
    //
    // offset = [3, 7, 10]
    //
    // MED:
    //
    // INN = [1, 4, 8, 11]
    //
    let mut inn = Vec::with_capacity(poly_conn.offsets.len() + 1);

    inn.push(1u64);

    for &end in poly_conn.offsets.iter() {
        inn.push(end as u64 + 1);
    }

    let inn_ds = group.new_dataset_builder().with_data(&inn).create("INN")?;

    write_scalar_attr(&inn_ds, "CGT", 1i64)?;
    write_scalar_attr(&inn_ds, "NBR", inn.len() as i64)?;

    Ok(())
}

fn write_polyhedron(mai: &Group, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    let poly_conn = match &block.connectivity {
        ConnectivityView::Poly(pc) => pc,
        _ => panic!("A PHED block must contain a poly connectivity."),
    };
    let group = mai.create_group("POE")?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;

    // Mefikit data layout (per element):
    //   [face0_n0, face0_n1, ..., usize::MAX, face1_n0, ..., usize::MAX, ...]
    //
    // MED POE layout:
    //   NOD = [face0_n0, face0_n1, ..., face1_n0, ...]  (1-based, flat)
    //   INN = [1, end_of_face0+1, end_of_face1+1, ...]  (1-based cumulative-end, n_faces+1)
    //   IFN = [1, end_of_face0_of_poly0+1, ...]         (1-based cumulative-end, n_poly+1)

    let mut nod: Vec<u64> = Vec::new();
    let mut inn: Vec<u64> = vec![1]; // leading sentinel
    let mut ifn: Vec<u64> = vec![1]; // leading sentinel

    // let n_poly = poly_conn.offsets.len();

    for elem_conn in poly_conn {
        // Split element connectivity by usize::MAX to get faces.
        let mut face_count = 0;
        for run in elem_conn.split(|&v| v == usize::MAX) {
            if run.is_empty() {
                continue;
            }
            // Convert 0-based node IDs to 1-based for MED.
            nod.extend(run.iter().map(|&n| (n + 1) as u64));
            // INN tracks cumulative end position in NOD (1-based).
            inn.push(nod.len() as u64 + 1);
            face_count += 1;
        }
        // IFN tracks cumulative end position in faces (1-based).
        ifn.push(ifn.last().unwrap() + face_count as u64);
    }

    let nod_ds = group.new_dataset_builder().with_data(&nod).create("NOD")?;
    write_scalar_attr(&nod_ds, "CGT", 1i64)?;
    write_scalar_attr(&nod_ds, "NBR", nod.len() as i64)?;

    let inn_ds = group.new_dataset_builder().with_data(&inn).create("INN")?;
    write_scalar_attr(&inn_ds, "CGT", 1i64)?;
    write_scalar_attr(&inn_ds, "NBR", inn.len() as i64)?;

    let ifn_ds = group.new_dataset_builder().with_data(&ifn).create("IFN")?;
    write_scalar_attr(&ifn_ds, "CGT", 1i64)?;
    write_scalar_attr(&ifn_ds, "NBR", ifn.len() as i64)?;

    Ok(())
}

fn write_block(mai: &Group, et: &ElementType, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    match et.regularity() {
        Regularity::Regular => {
            write_regular(mai, *et, block)?;
        }

        Regularity::Poly => match et {
            ElementType::PGON => {
                write_polygon(mai, block)?;
            }

            ElementType::PHED => {
                write_polyhedron(mai, block)?;
            }

            _ => {
                return Err(hdf5_metno::Error::Internal(format!(
                    "PolyConnectivity is not supported \
                                for {:?}",
                    et
                )));
            }
        },
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

fn med_name_to_element_type(name: &str) -> Option<ElementType> {
    match name {
        "PO1" => Some(ElementType::VERTEX),
        "SE2" => Some(ElementType::SEG2),
        "SE3" => Some(ElementType::SEG3),
        "TR3" => Some(ElementType::TRI3),
        "TR6" => Some(ElementType::TRI6),
        "TR7" => Some(ElementType::TRI7),
        "QU4" => Some(ElementType::QUAD4),
        "QU8" => Some(ElementType::QUAD8),
        "QU9" => Some(ElementType::QUAD9),
        "TE4" => Some(ElementType::TET4),
        "T10" => Some(ElementType::TET10),
        "HE8" => Some(ElementType::HEX8),
        "POG" | "POG2" => Some(ElementType::PGON),
        "POE" => Some(ElementType::PHED),
        _ => None,
    }
}

fn read_scalar_attr_i64(
    loc: &hdf5_metno::Location,
    name: &str,
) -> Result<i64, Box<dyn std::error::Error>> {
    let val = loc.attr(name)?.read_scalar::<i64>()?;
    Ok(val)
}

type Fas = (BTreeMap<i64, Vec<String>>, BTreeMap<i64, Vec<String>>);

/// Parse FAS/<mesh_name>/NOEUD and FAS/<mesh_name>/ELEME into
/// { family_id → [group_name, ...] }.
fn read_fas(fas_root: &Group, mesh_name: &str) -> Result<Fas, Box<dyn std::error::Error>> {
    let mesh_fas = match fas_root.group(mesh_name) {
        Ok(g) => g,
        Err(_) => return Ok((BTreeMap::new(), BTreeMap::new())),
    };

    let node_fams = read_fas_subgroup(&mesh_fas, "NOEUD");
    let elem_fams = read_fas_subgroup(&mesh_fas, "ELEME");

    Ok((node_fams, elem_fams))
}

fn read_fas_subgroup(parent: &Group, name: &str) -> BTreeMap<i64, Vec<String>> {
    let sub = match parent.group(name) {
        Ok(g) => g,
        Err(_) => return BTreeMap::new(),
    };

    let mut result = BTreeMap::new();

    for entry_name in sub.member_names().unwrap_or_default() {
        let fam_group = match sub.group(&entry_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let num = match fam_group.attr("NUM") {
            Ok(a) => a.read_scalar::<i64>().unwrap_or(0),
            Err(_) => continue,
        };

        let mut names = Vec::new();

        if let Ok(gro) = fam_group.group("GRO")
            && let Ok(nom_ds) = gro.dataset("NOM")
        {
            // NOM is an array of 80-byte char slots.
            let raw: Array2<i8> = nom_ds.read().unwrap_or_else(|_| Array2::zeros((0, 80)));
            for row in raw.rows() {
                let name_str: String = row
                    .iter()
                    .map(|&b| b as u8)
                    .take_while(|&b| b != 0 && b != b' ')
                    .map(|b| b as char)
                    .collect();
                if !name_str.is_empty() {
                    names.push(name_str);
                }
            }
        }

        result.insert(num, names);
    }

    result
}

fn read_nodal_data(
    timestep: &Group,
    dim: usize,
) -> Result<(usize, Array2<f64>), Box<dyn std::error::Error>> {
    let noe = timestep.group("NOE")?;
    let coo: Array1<f64> = noe.dataset("COO")?.read()?;
    let n_points = coo.len() / dim;
    // MED stores COO column-major: [x0,x1,...,xN, y0,y1,...,yN, ...]
    // Reshape as (dim, n_points) then transpose to get (n_points, dim).
    let raw = Array2::from_shape_vec((dim, n_points), coo.to_vec())
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let coords = raw.reversed_axes().as_standard_layout().to_owned();
    Ok((n_points, coords))
}

type RegBlock = (Array2<usize>, Option<Array1<i64>>);

fn read_regular_block(
    mai: &Group,
    med_type_name: &str,
    et: ElementType,
) -> Result<RegBlock, Box<dyn std::error::Error>> {
    let grp = mai.group(med_type_name)?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let n_nodes = et
        .num_nodes()
        .ok_or("element type has variable node count")?;
    let n_cells = nod.len() / n_nodes;

    // Column-major reshape, then 1-based → 0-based.
    let flat: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED stores (n_cells, n_nodes) in column-major: reshape as (n_nodes, n_cells) then transpose.
    let conn = Array2::from_shape_vec((n_nodes, n_cells), flat)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let conn = conn.reversed_axes().as_standard_layout().to_owned();

    // Apply inverse MED node permutation.
    let conn = match et.med_permutation() {
        Some(perm) => {
            let mut out = Array2::zeros((conn.nrows(), perm.len()));
            for (new_j, &old_j) in perm.iter().enumerate() {
                out.column_mut(new_j).assign(&conn.column(old_j));
            }
            out
        }
        None => conn,
    };

    let fam = read_fam_array(&grp)?;

    Ok((conn, fam))
}

fn read_fam_array(grp: &Group) -> Result<Option<Array1<i64>>, Box<dyn std::error::Error>> {
    Ok(grp.dataset("FAM").ok().and_then(|ds| ds.read().ok()))
}

type PolyBlock = (Array1<usize>, Array1<usize>, Option<Array1<i64>>);

fn read_polygon_block(
    mai: &Group,
    med_type_name: &str,
) -> Result<PolyBlock, Box<dyn std::error::Error>> {
    let grp = mai.group(med_type_name)?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let inn: Array1<i64> = grp.dataset("INN")?.read()?;

    // 1-based → 0-based.
    let data: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED INN has n+1 entries for n polygons: [1, end1+1, ..., endN+1].
    // Subtract 1 then drop leading 0 to get mefikit's n-entry cumulative-end offsets.
    let offsets: Vec<usize> = inn.iter().skip(1).map(|&x| (x - 1) as usize).collect();

    let fam = read_fam_array(&grp)?;

    Ok((Array1::from_vec(data), Array1::from_vec(offsets), fam))
}

fn read_polyhedron_block(mai: &Group) -> Result<PolyBlock, Box<dyn std::error::Error>> {
    let grp = mai.group("POE")?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let inn: Array1<i64> = grp.dataset("INN")?.read()?;
    let ifn: Array1<i64> = grp.dataset("IFN")?.read()?;

    // 1-based → 0-based for all three arrays.
    let nod: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED INN has n_faces+1 entries (leading 1-based sentinel). Convert to 0-based cumulative.
    let inn: Vec<usize> = inn.iter().map(|&x| (x - 1) as usize).collect();
    // MED IFN has n_poly+1 entries (leading 1-based sentinel). Convert to 0-based cumulative.
    let ifn: Vec<usize> = ifn.iter().map(|&x| (x - 1) as usize).collect();

    let n_poly = ifn.len() - 1;

    // Flatten each polyhedron's faces into a single contiguous slice,
    // inserting usize::MAX sentinels between faces (mefikit convention).
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(n_poly);

    for p in 0..n_poly {
        let face_start = ifn[p];
        let face_end = ifn[p + 1];
        for f_idx in face_start..face_end {
            let node_start = inn[f_idx];
            let node_end = inn[f_idx + 1];
            data.extend_from_slice(&nod[node_start..node_end]);
            data.push(usize::MAX);
        }
        offsets.push(data.len());
    }

    let fam = read_fam_array(&grp)?;

    Ok((Array1::from_vec(data), Array1::from_vec(offsets), fam))
}

/// Build group_name → element_index map from FAS family data and per-element FAM array.
fn build_groups_from_fas(
    fam_id_to_names: &BTreeMap<i64, Vec<String>>,
    fam_array: &Array1<i64>,
) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for (elem_idx, &fam_id) in fam_array.iter().enumerate() {
        if fam_id == 0 {
            continue;
        }
        if let Some(names) = fam_id_to_names.get(&fam_id) {
            for name in names {
                groups.entry(name.clone()).or_default().push(elem_idx);
            }
        }
    }

    groups
}

/// Assign families/groups to a block and insert it into the mesh.
fn insert_block(
    mesh: &mut UMesh,
    mut block: ElementBlock,
    fam: &Option<Array1<i64>>,
    elem_fam_map: &BTreeMap<i64, Vec<String>>,
) {
    if let Some(fam_arr) = fam {
        let groups = build_groups_from_fas(elem_fam_map, fam_arr);
        block.set_families(fam_arr.mapv(|x| x as usize).into_shared());
        if !groups.is_empty() {
            block.set_groups_internal(groups);
        }
    }
    let (key, wrapped) = block.into_entry();
    mesh.element_blocks.entry(key).or_insert(wrapped);
}

/// Read a MED file into a `UMesh`.
pub fn read(path: impl AsRef<Path>) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;

    // Navigate to mesh ensemble.
    let ensemble = file.group("ENS_MAA")?;
    let mesh_names = ensemble.member_names()?;
    if mesh_names.is_empty() {
        return Err("No meshes found in MED file".into());
    }
    if mesh_names.len() > 1 {
        return Err(format!(
            "MED file contains {} meshes; only single-mesh files are supported",
            mesh_names.len()
        )
        .into());
    }
    let mesh_name = &mesh_names[0];
    let med_mesh = ensemble.group(mesh_name)?;

    // Spatial dimension.
    let dim = read_scalar_attr_i64(&med_mesh, "ESP")? as usize;

    // Navigate to time-step level. If NOE is not directly under med_mesh,
    // descend into the sole time-step sub-group.
    let data_level = if med_mesh.group("NOE").is_ok() {
        med_mesh.clone()
    } else {
        let steps = med_mesh.member_names()?;
        if steps.is_empty() {
            return Err("No time-step groups found in mesh".into());
        }
        if steps.len() > 1 {
            return Err("Multiple time-step groups found; only single step is supported".into());
        }
        med_mesh.group(&steps[0])?
    };

    // Read coordinates.
    let (_n_points, coords) = read_nodal_data(&data_level, dim)?;

    let mut mesh = UMesh::new(coords.into_shared());

    // Parse FAS for family/group information.
    let fas_root = file.group("FAS").ok();
    let (_node_fam_map, elem_fam_map) = match &fas_root {
        Some(root) => read_fas(root, mesh_name)?,
        None => (BTreeMap::new(), BTreeMap::new()),
    };

    // Node families (stored in point_data convention; we skip for now since
    // mefikit doesn't have a direct per-node family slot on UMesh itself).

    // Read element blocks.
    let mai = data_level.group("MAI")?;

    for med_type_name in mai.member_names()? {
        let et = match med_name_to_element_type(&med_type_name) {
            Some(et) => et,
            None => continue, // Skip unknown element types.
        };

        match et {
            ElementType::PGON => {
                let (data, offsets, fam) = read_polygon_block(&mai, &med_type_name)?;
                let block = ElementBlock::new_poly(et, data.into_shared(), offsets.into_shared());
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
            ElementType::PHED => {
                let (data, offsets, fam) = read_polyhedron_block(&mai)?;
                let block = ElementBlock::new_poly(et, data.into_shared(), offsets.into_shared());
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
            _ => {
                let (conn, fam) = read_regular_block(&mai, &med_type_name, et)?;
                let block = ElementBlock::new_regular(et, conn.into_shared(), None, None);
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementLike;
    use crate::mesh_examples as me;
    use std::path::PathBuf;

    fn assert_mesh_eq(a: &UMesh, b: &UMesh) {
        assert_eq!(a.coords(), b.coords());
        let elems1: Vec<_> = a
            .elements()
            .map(|e| (e.element_type(), e.connectivity.to_vec()))
            .collect();
        let elems2: Vec<_> = b
            .elements()
            .map(|e| (e.element_type(), e.connectivity.to_vec()))
            .collect();
        assert_eq!(elems1.len(), elems2.len());
        for ((et1, c1), (et2, c2)) in elems1.iter().zip(elems2.iter()) {
            assert_eq!(et1, et2);
            assert_eq!(c1, c2);
        }
    }

    #[test]
    fn test_roundtrip_med_2d() {
        let path = PathBuf::from("test_roundtrip_med_2d.med");
        let mesh = me::make_mesh_2d_multi();
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_3d() {
        let path = PathBuf::from("test_roundtrip_med_3d.med");
        let mesh = me::make_imesh_3d(2);
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_phed() {
        let path = PathBuf::from("test_roundtrip_med_phed.med");

        let coords = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());

        let conn: Vec<usize> = vec![
            0,
            1,
            2,
            3,
            usize::MAX,
            4,
            7,
            6,
            5,
            usize::MAX,
            0,
            4,
            5,
            1,
            usize::MAX,
            2,
            6,
            7,
            3,
            usize::MAX,
            0,
            3,
            7,
            4,
            usize::MAX,
            1,
            5,
            6,
            2,
            usize::MAX,
        ];
        mesh.add_element(ElementType::PHED, &conn, None, None);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }
}

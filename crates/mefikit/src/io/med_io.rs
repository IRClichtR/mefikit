use crate::mesh::ConnectivityView;
use crate::mesh::ElementBlockView;
use crate::mesh::ElementType;
use crate::mesh::Regularity;
use crate::mesh::UMeshView;

use hdf5_metno::types::FixedAscii;
use hdf5_metno::types::VarLenAscii;
use hdf5_metno::{File, Group};
use ndarray::prelude::*;

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
    let med_conn = dbg!(reorder_connectivity(
        dbg!(conn),
        element_type.med_permutation()
    ));

    let mut flat = Vec::with_capacity(med_conn.len());

    // MED expects column-major ordering.
    for j in 0..med_conn.ncols() {
        for i in 0..med_conn.nrows() {
            flat.push(med_conn[[i, j]] + 1);
        }
    }

    let nod = group
        .new_dataset_builder()
        .with_data(&dbg!(flat))
        .create("NOD")?;

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

fn write_block(mai: &Group, et: &ElementType, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    match et.regularity() {
        Regularity::Regular => {
            write_regular(mai, *et, block)?;
        }

        Regularity::Poly => match et {
            ElementType::PGON => {
                write_polygon(mai, block)?;
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
